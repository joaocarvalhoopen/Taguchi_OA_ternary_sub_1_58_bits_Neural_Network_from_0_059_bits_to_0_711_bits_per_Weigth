// Taguchi OA ternary sub 1.58 bits Neural Network from 0.059_bits to 0.711 bits per Weigth
//
// High compression Weight encoded training with Orthogonal Arrays,
// Taguchi of ternary -1, 0, 1 weights of 1.58 bits style Neural Network.
//
// Description
// - OA - Orthogonal Arrays / Taguchi ternary coefficient layers + STE-style training
// - Minimal model save : packed trits (5 trits per byte) + int16 scale/bias
// - Parallel training : 24 threads via pthreads
//
// MNIST folder must contain uncompressed IDX files:
//   train-images-idx3-ubyte
//   train-labels-idx1-ubyte
//   t10k-images-idx3-ubyte
//   t10k-labels-idx1-ubyte
//
// Build:
//   odin build . -out:taguchi_ternary_mnist_neural_network.exe -o:speed -no-bounds-check
//
// Run:
//   ./taguchi_ternary_mnist_neural_network.exe train <mnist_folder> <model.bin> [epochs]
//   ./taguchi_ternary_mnist_neural_network.exe <mnist_folder> <model.bin> <train|test> <index>
//


package main

import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:time"
import "core:math"
import "core:slice"
import "core:strconv"
import rand "core:math/rand"
import sync "core:sync"
import posix "core:sys/posix"

//
// Config
//

NUM_THREADS :: 24

LR_START    :: 3e-2
LR_END      :: 1e-5

// Neural Network Stucture

MNIST_ROWS  :: 28
MNIST_COLS  :: 28
INPUT_DIM   :: MNIST_ROWS * MNIST_COLS

// H1       :: 256
// H2       :: 128
// OUT      :: 10

H1          :: 400
H2          :: 128
OUT         :: 10


DEFAULT_BATCH  :: 512
DEFAULT_EPOCHS :: 40

//    0.711  Bits / Weight
//
// TOP MNIST Test_accuracy : 96.17%
//
//  Epoch 244/300 | loss 0.0011 | test acc 96.17% | lr 4.45739e-05
// DEFAULT_OA_M   :: 2
// K_USED_DEFAULT :: 4

// Very good result       0.533  Bits / Weight
//
// TOP MNIST Test_accuracy : 96.41 %
//
//   <---- GOOD  Test_Accuracy 95.87 %  -> 96.09 % -> epoch=154 96.20 %    epoch=171 96.23 %, epoch=185  96.32 %, epoch=212  96.35 % epoch 236  96.41 %
DEFAULT_OA_M   :: 2
K_USED_DEFAULT :: 3


// Very small weights file     0.178  Bits / Weight
//
// TOP MNIST Test_accuracy :
//
//     <----- Epoch=208  69.58 %,  Epoch=237 69.79 % Epoch=300 loss_0.9797 test_acc_70.05 %
// DEFAULT_OA_M   :: 2
// K_USED_DEFAULT :: 1



// TERNARY_ALPHA :: 0.7
TERNARY_ALPHA    :: 0.5

ADAM_BETA1       :: 0.9
ADAM_BETA2       :: 0.999
ADAM_EPS         :: 1e-8
WEIGHT_DECAY     :: 0.0

// Taguchi Compressed Model
MODEL_MAGIC      :: "TGCM"
MODEL_VERSION    :: 4

//
// Utils
//

fatal :: proc ( msg  : string,
                args : ..any ) ->
                ! {

	fmt.printf( "FATAL: " )
    fmt.printfln( msg, ..args )

	os.exit( -1 )
}

chunk_range :: proc "contextless" (
                      n     : int,
                      tid   : int,
                      nt    : int,
                      start : ^int,
                      end_  : ^int ) {

	base  := n / nt
	rem   := n % nt
	s     := tid * base + ( tid < rem ? tid : rem )
	e     := s + base + ( tid < rem ? 1 : 0 )
	start^ = s
	end_^  = e
}

ipow3 :: proc "contextless" ( m : int ) ->
                              int {

	r := 1
	for i in 0 ..< m {

		r *= 3
	}
	return r
}

vec_to_code_base3 :: proc "contextless" (
                           v : [ ]u8,
                           m : int ) ->
                           int {

	code := 0
	p    := 1

	for i in 0 ..< m {

		code += int( v[ i ] ) * p
		p *= 3
	}

	return code
}

code_to_vec_base3 :: proc "contextless" (
                                    code : int,
                                    v    : [ ]u8,
                                    m    : int ) {

	c := code
	for i in 0 ..< m {

		v[ i ] = u8( c % 3 )
		c /= 3
	}
}

first_nonzero_index :: proc "contextless" (
                                        v : [ ]u8,
                                        m : int ) ->
                                        int {

	for i in 0 ..< m {

		if v[ i ] != 0 {

			return i
		}
	}

	return -1
}

to_float_pixel :: proc "contextless" ( p : u8 ) ->
                                       f32 {

    // Center the data around zero.
    // NOTE : This gives good results.
	return f32( p ) / 255.0 - 0.5


	// MNIST mean / std normalization

	// mean +-= 0.1307
    // std  +-= 0.3081

    // x := f32( p ) / 255.0
    // return ( x - 0.1307 ) / 0.3081
}

//
// Endian IO
//

// MNIST IDX is big-endian u32.
read_be_u32 :: proc ( fd : os.Handle ) ->
                      u32 {

	buf : [ 4 ]u8
	n, err := os.read_full( fd, buf[ : ] )
	if err != os.ERROR_NONE || n != 4 {

		fatal( "failed to read be u32" )
	}
	return ( u32( buf[ 0 ] ) << 24 ) |
           ( u32( buf[ 1 ] ) << 16 ) |
           ( u32( buf[ 2 ] ) << 8 )  |
           u32( buf[ 3 ] )
}

// Model file is little-endian
write_u32_le :: proc ( fd : os.Handle,
                       v  : u32 ) {

	b : [ 4 ]u8
	b[ 0 ] = u8( v & 0xFF )
	b[ 1 ] = u8( ( v >> 8 ) & 0xFF )
	b[ 2 ] = u8( ( v >> 16 ) & 0xFF )
	b[ 3 ] = u8( ( v >> 24 ) & 0xFF )
	n, err := os.write( fd, b[ : ] )
	if err != os.ERROR_NONE || n != 4 {

		fatal( "failed to write u32" )
	}
}

read_u32_le :: proc ( fd: os.Handle ) ->
                      u32 {

	b : [ 4 ]u8
	n, err := os.read_full( fd, b[ : ] )
	if err != os.ERROR_NONE || n != 4 {

		fatal( "failed to read u32" )
	}

	return u32( b[ 0 ] ) |
           ( u32( b[ 1 ] ) << 8 ) |
           ( u32( b[ 2 ] ) << 16 ) |
           ( u32( b[ 3 ] ) << 24 )
}

write_i16_le :: proc ( fd : os.Handle,
                       v  : i16 ) {

	u := u16( v )
	b : [ 2 ]u8
	b[ 0 ] = u8( u & 0xFF )
	b[ 1 ] = u8( ( u >> 8 ) & 0xFF )
	n, err := os.write( fd, b[ : ] )
	if err != os.ERROR_NONE || n != 2 {

		fatal( "failed to write i16" )
	}
}

read_i16_le :: proc ( fd : os.Handle ) -> i16 {

	b : [ 2 ]u8
	n, err := os.read_full( fd, b[ : ] )
	if err != os.ERROR_NONE || n != 2 {

		fatal( "failed to read i16" )
	}
	u := u16( b[ 0 ] ) |
         ( u16( b[ 1 ] ) << 8 )

	return i16( u )
}

write_f32_le :: proc ( fd : os.Handle,
                       x  : f32 ) {

	u := transmute( u32 )( x )
	write_u32_le( fd, u )
}

read_f32_le :: proc ( fd : os.Handle ) ->
                      f32 {

	u := read_u32_le( fd )
	return transmute( f32 )u
}

//
// Random functions
//

frand01 :: proc ( ) ->
                 f32 {

	// rand.float32( ) in [ 0, 1 )
	return rand.float32( )
}

frandn :: proc( ) ->
               f32 {

	// Box–Muller transform (same as C code behavior)
	u1 := frand01( )
	u2 := frand01( )
	if u1 < 1e-12 {

		u1 = 1e-12
	}
	r := math.sqrt( -2.0 * math.ln( f64( u1 ) ) )
	theta := 2.0 * math.PI * f64( u2 )
	return f32( r * math.cos( theta ) )
}

//
// Thread pool
//

Par_Job_Fn :: proc ( tid : int, nt : int, ctx : rawptr )

// #no_copy

Thread_Pool :: struct  {

	threads     : [ NUM_THREADS ]posix.pthread_t,
	worker_args : [ NUM_THREADS ]Worker_Arg,

	mtx         : sync.Mutex,
	cv_job      : sync.Cond,
	cv_done     : sync.Cond,

	fn          : Par_Job_Fn,
	ctx         : rawptr,
	epoch       : i32,
	pending     : i32,
	stop        : bool,
}

Worker_Arg :: struct {

	pool : ^Thread_Pool,
	tid  : int,
}

worker_main :: proc "c" ( arg : rawptr ) ->
                          rawptr {

    context = runtime.default_context()

	wa  := ( ^Worker_Arg )( arg )
	p   := wa.pool
	tid := wa.tid

	local_epoch : i32 = 0

	sync.mutex_lock( & p.mtx )
	for {

		for !p.stop && p.epoch == local_epoch {

			sync.cond_wait( & p.cv_job, & p.mtx )
		}
		if p.stop {

			break
		}

		local_epoch = p.epoch
		fn         := p.fn
		ctx        := p.ctx

		sync.mutex_unlock( & p.mtx )
		fn( tid, NUM_THREADS, ctx )
		sync.mutex_lock( & p.mtx )

		p.pending -= 1
		if p.pending == 0 {

			sync.cond_signal( & p.cv_done )
		}
	}
	sync.mutex_unlock( & p.mtx )
	return nil
}

pool_init :: proc( p : ^Thread_Pool ) {

	mem.zero( p, size_of( p^ ) )
	p.epoch   = 0
	p.pending = 0
	p.stop    = false

	for i in 0 ..< NUM_THREADS {

		p.worker_args[ i ].pool = p
		p.worker_args[ i ].tid  = i

		// pthread_create( & thread, nil, worker_main, & arg )
		rc := posix.pthread_create( & p.threads[ i ], nil, worker_main, & p.worker_args[ i ] )
		if rc != .NONE {

			fatal( "pthread_create failed ( rc=%d )", rc )
		}
	}
}

pool_run :: proc ( p   : ^Thread_Pool,
                   fn  : Par_Job_Fn,
                   ctx : rawptr) {

	sync.mutex_lock( & p.mtx )
	p.fn      = fn
	p.ctx     = ctx
	p.pending = NUM_THREADS
	p.epoch  += 1
	sync.cond_broadcast( & p.cv_job )
	for p.pending > 0 {

		sync.cond_wait( & p.cv_done, & p.mtx )
	}
	sync.mutex_unlock( & p.mtx )
}

pool_destroy :: proc ( p : ^Thread_Pool ) {

	sync.mutex_lock( & p.mtx )
	p.stop   = true
	p.epoch += 1
	sync.cond_broadcast( & p.cv_job )
	sync.mutex_unlock( & p.mtx )

	for i in 0 ..< NUM_THREADS {

		_ = posix.pthread_join( p.threads[ i ], nil )
	}
}

//
// MNIST loading
//

MNIST :: struct {

    count  : int,
	rows   : int,
	cols   : int,
	images : [ ]u8,
	labels : [ ]u8,
}

load_mnist :: proc ( images_path : string,
                     labels_path : string ) ->
                     MNIST {

	d : MNIST

	// labels
	fl, err := os.open( labels_path, os.O_RDONLY, 0 )
	if err != os.ERROR_NONE {

		fatal( "cannot open labels: %s", labels_path )
	}
	defer os.close( fl )

	magic_l := read_be_u32( fl )
	count_l := read_be_u32( fl )
	if magic_l != 2049 {

		fatal( "labels magic mismatch" )
	}

	d.labels = make( [ ]u8, int( count_l ) )
	nl, err2 := os.read_full( fl, d.labels )
	if err2 != os.ERROR_NONE || nl != len( d.labels ) {

		fatal( "labels read failed" )
	}

	// images
	fi, err3 := os.open( images_path, os.O_RDONLY, 0 )
	if err3 != os.ERROR_NONE {

		fatal( "cannot open images: %s", images_path )
	}
	defer os.close( fi )

	magic_i := read_be_u32( fi )
	count_i := read_be_u32( fi )
	rows    := read_be_u32( fi )
	cols    := read_be_u32( fi )

	if magic_i != 2051 {

		fatal( "images magic mismatch" )
	}
	if count_i != count_l {

		fatal( "count mismatch ( images %d labels %d )", count_i, count_l )
	}
	if int( rows ) != MNIST_ROWS || int( cols ) != MNIST_COLS {

		fatal( "expected 28x28 images" )
	}

	img_bytes := int( count_i ) * int( rows ) * int( cols )
	d.images   = make( [ ]u8, img_bytes )

	ni, err4 := os.read_full( fi, d.images )
	if err4 != os.ERROR_NONE || ni != len( d.images ) {

		fatal( "images read failed" )
	}

	d.count = int( count_i )
	d.rows  = int( rows )
	d.cols  = int( cols )

	return d
}

free_mnist :: proc ( d : ^MNIST ) {

	if len( d.images ) > 0 {

	    delete( d.images )
	}

	if len( d.labels ) > 0 {

	    delete( d.labels )
	}

	mem.zero( d, size_of( d^ ) )
}

//
//  Taguchi/OA basis - Orthogonal Arrays basis builder
//

OA_Basis :: struct {

	m : int,
	N : int,
	K : int,
	A : [ ]i8, // N * K in {-1, 0, +1 }, row-major
}

build_oa_basis_full :: proc ( m : int ) ->
                              OA_Basis {

	b : OA_Basis
	b.m = m
	b.N = ipow3( m )
	b.K = ( b.N - 1 ) / 2

	// X : all base-3 vectors of length m
	X := make( [ ]u8, b.N * m )
	defer delete( X )

	for r in 0 ..< b.N {

		code_to_vec_base3( r, X[ r * m : ( r + 1 ) * m ], m )
	}

	seen := make( [ ]u8, b.N )
	defer delete( seen )

	reps := make( [ ]u8, b.K * m )
	rep_count := 0

	tmp  : [ 16 ]u8
	tmp2 : [ 16 ]u8

	for code in 1 ..< b.N {

		code_to_vec_base3( code, tmp[ : m ], m )
		idx := first_nonzero_index( tmp[ : m ], m )
		if idx < 0 {

			continue
		}

		inv : u8 = 2
		if tmp[ idx ] == 1 {

			inv = 1
		}
		for j in 0 ..< m {

		    tmp[ j ] = u8( ( u32( inv ) * u32( tmp[ j ] ) ) % 3 )
		}

		norm_code := vec_to_code_base3( tmp[ : m ], m )
		if seen[ norm_code ] != 0 {

			continue
		}

		for j in 0 ..< m {

		    tmp2[ j ] = u8( ( 2 * u32( tmp[ j ] ) ) % 3 )
		}
		code2 := vec_to_code_base3( tmp2[ : m ], m )

		seen[ norm_code ] = 1
		seen[ code2 ]     = 1

		copy( reps[ rep_count * m : ( rep_count + 1 ) * m ], tmp[ : m ] )
		rep_count += 1
	}

	if rep_count != b.K {

		fatal( "OA rep_count mismatch ( %d vs %d )", rep_count, b.K )
	}

	b.A = make( [ ]i8, b.N * b.K )

	for r in 0 ..< b.N {

		xr := X[ r * m : ( r + 1 ) * m ]
		for k in 0 ..< b.K {

		    ak := reps[ k * m : ( k + 1 ) * m ]
			dot := 0
			for j in 0 ..< m {

				dot += int( xr[ j ] ) * int( ak[ j ] )
			}

			dot %= 3
			b.A[ r * b.K + k ] = i8( dot - 1 )     // { -1, 0, 1 }
		}
	}

	return b
}

trim_basis :: proc ( full   : ^OA_Basis,
                     K_used : int ) ->
                     OA_Basis {

	if K_used <= 0 || K_used > full.K {

		fatal( "invalid K_used" )
	}

	t   : OA_Basis
	t.m = full.m
	t.N = full.N
	t.K = K_used
	t.A = make( [ ]i8, t.N * t.K )

	for n in 0 ..< t.N {

		src := full.A[ n * full.K : n * full.K + t.K ]
		dst := t.A[ n * t.K : ( n + 1 ) * t.K ]
		copy( dst, src )
	}

	return t
}

free_oa_basis :: proc ( b : ^OA_Basis ) {

	if len( b.A ) > 0 {

	    delete( b.A )
	}
	mem.zero( b, size_of( b^ ) )
}

//
// Trit packing ( base-3 ) kind of ternary bits :-)
//

trit_to_digit :: #force_inline proc "contextless" ( t : i8 ) ->
                                                    u8 {

	if t < 0 {

	    return 0
	}

	if t == 0 {

	    return 1
	}

	return 2
}

digit_to_trit :: #force_inline proc "contextless" ( d : u8 ) ->
                                                    i8 {

	if d == 0 {

	    return -1
	}

	if d == 1 {

	    return 0
	}

	return 1
}

packed_bytes_for_trits :: #force_inline proc "contextless" ( trit_count : int ) ->
                                                             int {

	return ( trit_count + 4 ) / 5
}

pack_trits_5perbyte :: proc( trits     : [ ]i8,
                             out_bytes : [ ]u8 ) {

	n_trits := len( trits )
	bi := 0
	i  := 0
	for i < n_trits {

		val : u32 = 0
		p   : u32 = 1
		for j in 0 ..< 5 {

			d : u8 = 1
			if i < n_trits {

			    d = trit_to_digit( trits[ i ] )
				i += 1
			}
			val += u32( d ) * p
			p *= 3
		}

		out_bytes[ bi ] = u8( val )
		bi += 1
	}
}

unpack_trits_5perbyte :: proc ( bytes     : [ ]u8,
                                n_trits   : int,
                                out_trits : [ ]i8 ) {

	bi := 0
	i  := 0
	for i < n_trits {

		val := u32( bytes[ bi ] )
		bi += 1
		for j := 0; j < 5 && i < n_trits; j += 1 {

			d := u8( val % 3 )
			val /= 3
			out_trits[ i ] = digit_to_trit( d )
			i += 1
		}
	}
}

//
// Quantize scale/bias to int16
//

max_abs :: proc ( a : [ ]f32 ) ->
                  f32 {

	m : f32 = 0
	for v in a {

		av := f32( math.abs( f64( v ) ) )
		if av > m {

	        m = av
	    }
	}
	return m
}

quantize_f32_to_i16 :: proc ( src       : [ ]f32,
                              qstep_out : ^f32,
                              dst_q     : [ ]i16 ) {

	m := max_abs( src )
	if m < 1e-12 {

		qstep_out^ = 1.0
		for i in 0 ..< len( dst_q ) {

		    dst_q[ i ] = 0
	    }

		return
	}

	qstep     := m / 32767.0
	qstep_out^ = qstep
	for i in 0 ..< len( src ) {

		qf := f64( src[ i ] / qstep )
		q := int( math.round( qf ))
		if q > 32767 {

		    q = 32767
	    }
		if q < -32767 {

		    q = -32767
	    }
		dst_q[ i ] = i16( q )
	}
}

dequantize_i16_to_f32 :: proc ( src_q : [ ]i16,
                                qstep : f32,
                                dst   : [ ]f32 ) {

	for i in 0 ..< len( dst ) {

		dst[ i ] = f32( src_q[ i ] ) * qstep
	}
}

//
// OA Linear ( training )
//

OA_Linear :: struct {

    in_dim    : int,
    out_dim   : int,
	m         : int,
    N         : int,
    K         : int,
	blocks    : int,
	inv_sqrtK : f32,
	alpha     : f32,
	basis     : [ ]i8,

	coef      : [ ]f32,
	scale     : [ ]f32,
	bias      : [ ]f32,

	m_coef  : [ ]f32,
    v_coef  : [ ]f32,
	m_scale : [ ]f32,
    v_scale : [ ]f32,
	m_bias  : [ ]f32,
    v_bias  : [ ]f32,

	g_coef  : [ ]f32,
	g_scale : [ ]f32,
	g_bias  : [ ]f32,

	cq      : [ ]i8,      // quantized coef in { -1, 0, 1 }
}

coef_size :: proc "contextless" ( L : ^OA_Linear ) ->
                                  int {

	return L.out_dim * L.blocks * L.K
}

oalinear_init :: proc ( L       : ^OA_Linear,
                        in_dim  : int,
                        out_dim : int,
                        B       : ^OA_Basis,
                        alpha   : f32 ) {

    mem.zero( L, size_of( L^ ) )
	L.in_dim  = in_dim
	L.out_dim = out_dim
	L.m = B.m
	L.N = B.N
	L.K = B.K
	L.blocks    = ( in_dim + L.N - 1 ) / L.N
	L.inv_sqrtK = 1.0 / f32( math.sqrt( f64( L.K ) ) )
	L.alpha = alpha
	L.basis = B.A

	cs := coef_size( L )

	L.coef  = make( [ ]f32, cs )
	L.scale = make( [ ]f32, out_dim )
	L.bias  = make( [ ]f32, out_dim )

	L.m_coef = make( [ ]f32, cs )
	L.v_coef = make( [ ]f32, cs )
	L.m_scale = make( [ ]f32, out_dim )
	L.v_scale = make( [ ]f32, out_dim )
	L.m_bias  = make( [ ]f32, out_dim )
	L.v_bias  = make( [ ]f32, out_dim )

	L.g_coef  = make( [ ]f32, cs )
	L.g_scale = make( [ ]f32, out_dim )
	L.g_bias  = make( [ ]f32, out_dim )

	L.cq = make( [ ]i8, cs )

	for i in 0 ..< cs {

		L.coef[ i ] = 0.02 * frandn( )
	}
	for o in 0 ..< out_dim {

		L.scale[ o ] = 1.0
		L.bias[ o ]  = 0.0
	}
}

oalinear_free :: proc ( L : ^OA_Linear ) {

	if len( L.coef ) > 0 {

	    delete( L.coef )
	}
	if len( L.scale ) > 0 {

	    delete( L.scale )
	}
	if len( L.bias ) > 0 {

	    delete( L.bias )
	}

	if len( L.m_coef ) > 0 {

	    delete( L.m_coef )
	}
	if len( L.v_coef ) > 0 {

	    delete( L.v_coef )
	}
	if len( L.m_scale ) > 0 {

	    delete( L.m_scale )
	}
	if len( L.v_scale ) > 0 {

	    delete( L.v_scale )
	}
	if len( L.m_bias ) > 0 {

	    delete( L.m_bias )
	}
	if len( L.v_bias ) > 0 {

	    delete( L.v_bias )
	}

	if len( L.g_coef ) > 0 {

	    delete( L.g_coef )
	}
	if len( L.g_scale ) > 0 {

	    delete( L.g_scale )
	}
	if len( L.g_bias ) > 0 {

	    delete( L.g_bias )
	}

	if len( L.cq ) > 0 {

	    delete( L.cq )
	}

	mem.zero( L, size_of( L^ ) )
}

//
// Parallel kernels
//

// Quantize_coef parallel
Quant_Ctx :: struct {

    L: ^OA_Linear
}

job_quantize :: proc ( tid  : int,
                       nt   : int,
                       ctxp : rawptr ) {

	ctx := ( ^Quant_Ctx )( ctxp )
	L := ctx.L

	groups := L.out_dim * L.blocks         // each group has K values
	gs : int
    ge : int
	chunk_range( groups, tid, nt, & gs, & ge )

	for g := gs; g < ge; g += 1 {

		base := g * L.K

		meanabs: f32 = 0

		for k in 0 ..< L.K {

			meanabs += f32( math.abs( f64( L.coef[ base + k ] ) ) )
		}
		meanabs /= f32( L.K )

		t := L.alpha * meanabs + 1e-8
		for k in 0 ..< L.K {

			v := L.coef[ base + k ]
			q : i8 = 0
			if v > t {

			    q = 1
			} else if v < -t {

			    q = -1
			}

			L.cq[ base + k ] = q
		}
	}
}

// Forward parallel ( partition by samples )
Fwd_Ctx :: struct {

	L     : ^OA_Linear,
	x     : [ ]f32,
	batch : int,
	y     : [ ]f32,
	pre   : [ ]f32,
}

job_forward :: proc ( tid  : int,
                      nt   : int,
                      ctxp : rawptr ) {

	ctx := ( ^Fwd_Ctx )( ctxp )
	L := ctx.L

	s0 : int
    s1 : int
	chunk_range( ctx.batch, tid, nt, & s0, & s1 )

	for s := s0; s < s1; s += 1 {

		xs := ctx.x[ s * L.in_dim : ( s + 1 ) * L.in_dim ]
		ys := ctx.y[ s * L.out_dim : ( s + 1 ) * L.out_dim ]
		ps := ctx.pre[ s * L.out_dim : ( s + 1 ) * L.out_dim ]

		for o in 0 ..< L.out_dim {

			acc : f32 = 0
			for i in 0 ..< L.in_dim {

			    block := i / L.N
				n := i - block * L.N

				base_row := n * L.K
				cq_off   := ( o * L.blocks + block ) * L.K

				dot := 0

				// aqui
				assert( L.K == K_USED_DEFAULT )
				for k in 0 ..< L.K {
				// #unroll for k in 0 ..< K_USED_DEFAULT {

					a := int( L.basis[base_row + k ] )
					c := int( L.cq[ cq_off + k ] )
					dot += a * c
				}
				acc += xs[ i ] * ( f32( dot ) * L.inv_sqrtK )
			}

			ps[ o ] = acc
			ys[ o ] = L.bias[ o ] + L.scale[ o ] * acc
		}
	}
}

//
// ReLU forward / backward parallel
//
Relu_Fwd_Ctx :: struct {

    x : [ ]f32,
    y : [ ]f32
}

job_relu_fwd :: proc ( tid  : int,
                       nt   : int,
                       ctxp : rawptr ) {

	ctx := ( ^Relu_Fwd_Ctx )( ctxp )
	n := len( ctx.x )
	s : int
    e : int
	chunk_range( n, tid, nt, & s, & e )
	for i := s; i < e; i += 1 {

		v := ctx.x[ i ]
		ctx.y[ i ] = ( v > 0 ) ? v : 0
	}
}

Relu_Bwd_Ctx :: struct {

    y_relu : [ ]f32,
    grad   : [ ]f32
}

job_relu_bwd :: proc ( tid  : int,
                       nt   : int,
                       ctxp : rawptr ) {

	ctx := ( ^Relu_Bwd_Ctx )( ctxp )
	n := len( ctx.grad )
	s : int
    e : int
	chunk_range( n, tid, nt, & s, & e )
	for i := s; i < e; i += 1 {

		if ctx.y_relu[ i ] <= 0 {

			ctx.grad[ i ] = 0
		}
	}
}

// Softmax + xent parallel ( by samples )
Softmax_Ctx :: struct {

	logits    : [ ]f32,             // [ batch, OUT ]
	labels    : [ ]u8,              // [ batch ]
	batch     : int,
	dlogits   : [ ]f32 ,            // [ batch, OUT ]
	loss_part : [ NUM_THREADS ]f32,
}

job_softmax_xent :: proc ( tid  : int,
                           nt   : int,
                           ctxp : rawptr ) {

	ctx := ( ^Softmax_Ctx )( ctxp )
	s0 : int
    s1 : int
	chunk_range( ctx.batch, tid, nt, & s0, & s1 )

	lsum : f32 = 0

	for s := s0; s < s1; s += 1 {

		ls := ctx.logits[ s * OUT : ( s + 1 ) * OUT ]
		gs := ctx.dlogits[ s * OUT : ( s + 1 ) * OUT ]
		y  := int( ctx.labels[ s ] )

		mx := ls[ 0 ]
		for j := 1; j < OUT; j += 1 {

			if ls[ j ] > mx {

			    mx = ls[ j ]
		    }
		}

		exps : [ OUT ]f32
		sum  : f32 = 0
		for j := 0; j < OUT; j += 1 {

		    exps[ j ] = f32( math.exp( f64( ls[ j ] - mx ) ) )
			sum += exps[ j ]
		}

		invsum := 1.0 / sum
		py     := exps[ y ] * invsum
		lsum   += -f32( math.ln( f64( py + 1e-12 ) ) )

		for j := 0; j < OUT; j += 1 {

			p := exps[ j ] * invsum
			g := p - ( ( j == y ) ? 1.0 : 0.0)
			gs[ j ] = g / f32( ctx.batch )
		}
	}

	ctx.loss_part[ tid ] = lsum
}

//
// Zero grads parallel (mostly g_coef)
//
Zero_Grad_Ctx :: struct {

    L : ^OA_Linear
}

job_zero_grads :: proc ( tid  : int,
                         nt   : int,
                         ctxp : rawptr ) {

    ctx := ( ^Zero_Grad_Ctx )( ctxp )
	L   := ctx.L
	cs  := coef_size( L )

	s : int
    e : int
	chunk_range( cs, tid, nt, & s, & e )
	for i := s; i < e; i += 1 {

		L.g_coef[ i ] = 0
	}

	if tid == 0 {

		for o in 0 ..< L.out_dim {

			L.g_scale[ o ] = 0
			L.g_bias[ o ]  = 0
		}
	}
}

// Backward : g_bias / g_scale parallel by outputs
GBias_Scale_Ctx :: struct {

	L     : ^OA_Linear,
	pre   : [ ]f32,      // [ batch, out ]
	dY    : [ ]f32,      // [ batch, out ]
	batch : int,
}

job_gbias_gscale :: proc ( tid  : int,
                           nt   : int,
                           ctxp : rawptr ) {

	ctx := ( ^GBias_Scale_Ctx )( ctxp )
	L   := ctx.L
	o0 : int
    o1 : int
	chunk_range( L.out_dim, tid, nt, & o0, & o1 )

	for o := o0; o < o1; o += 1 {

		gb : f32 = 0
		gs : f32 = 0
		for s in 0 ..< ctx.batch {

		    g := ctx.dY[ s * L.out_dim + o ]
			gb += g
			gs += g * ctx.pre[ s * L.out_dim + o ]
		}
		L.g_bias[ o ]  = gb
		L.g_scale[ o ] = gs
	}
}

// Backward: g_coef parallel by outputs
GCoef_Ctx :: struct {

    L     : ^OA_Linear,
	x     : [ ]f32,       // [ batch, in ]
	dY    : [ ]f32,       // [ batch, out ]
	batch : int,
}

job_gcoef :: proc ( tid  : int,
                    nt   : int,
                    ctxp : rawptr ) {

	ctx := ( ^GCoef_Ctx )( ctxp )
	L := ctx.L
	o0 : int
	o1 : int
	chunk_range( L.out_dim, tid, nt, & o0, & o1 )

	for o := o0; o < o1; o += 1 {

		so := L.scale[ o ]
		for block in 0 ..< L.blocks {

			i0 := block * L.N
			i1 := i0 + L.N
			if i1 > L.in_dim {

			    i1 = L.in_dim
		    }

			cq_off := ( o * L.blocks + block ) * L.K

			for k in 0 ..< L.K {

				acc : f32 = 0

				for s in 0 ..< ctx.batch {

				    g  := ctx.dY[ s * L.out_dim + o ] * so
					xs := ctx.x[ s * L.in_dim : ( s + 1 ) * L.in_dim ]

					for i := i0; i < i1; i += 1 {

						n := i - i0
						a := f32( int( L.basis[ n * L.K + k ]  ) )
						acc += g * xs[ i ] * a * L.inv_sqrtK
					}
				}

				// Unique per ( o, block, k) since we partition by o => no race
				L.g_coef[ cq_off + k ] += acc
			}
		}
	}
}

//
// backward: dX parallel by samples
//
DX_Ctx :: struct {

	L     : ^OA_Linear,
	dY    : [ ]f32,
	batch : int,
	dX    : [ ]f32,    // [ batch, in ]
}

job_dx :: proc ( tid  : int,
                 nt   : int,
                 ctxp : rawptr ) {

	ctx := ( ^DX_Ctx )( ctxp )
	L := ctx.L
	s0, s1: int
	chunk_range( ctx.batch, tid, nt, & s0, & s1 )

	for s := s0; s < s1; s += 1 {

		dxs := ctx.dX[ s * L.in_dim : ( s + 1 ) * L.in_dim ]

		for i := 0; i < L.in_dim; i += 1 {

			block    := i / L.N
			n        := i - block * L.N
			base_row := n * L.K

			acc : f32 = 0
			for o := 0; o < L.out_dim; o += 1 {

				so     := L.scale[ o ]
				cq_off := ( o * L.blocks + block ) * L.K

				dot := 0
				for k in 0 ..< L.K {

					a := int( L.basis[base_row + k ] )
					c := int( L.cq[ cq_off + k ] )
					dot += a * c
				}
				w := so * ( f32( dot ) * L.inv_sqrtK )
				acc += ctx.dY[ s * L.out_dim + o ] * w
			}

			dxs[ i ] = acc
		}
	}
}

// Adam update parallel for big arrays ( coef )
Adam_Ctx :: struct {

	param : [ ]f32,
	grad  : [ ]f32,
	m     : [ ]f32,
	v     : [ ]f32,
	n     : int,
	lr    : f32,
	beta1 : f32,
	beta2 : f32,
	eps   : f32,
	wd    : f32,
	b1t   : f32,
	b2t   : f32,
}

job_adam :: proc ( tid  : int,
                   nt   : int,
                   ctxp : rawptr ) {

	ctx := ( ^Adam_Ctx )( ctxp )
	s : int
    e : int
	chunk_range( ctx.n, tid, nt, & s, & e )

	for i := s; i < e; i += 1 {

		g := ctx.grad[ i ]
		if ctx.wd != 0 {

			g += ctx.wd * ctx.param[ i ]
		}

		ctx.m[ i ] = ctx.beta1 * ctx.m[ i ] + ( 1.0 - ctx.beta1 ) * g
		ctx.v[ i ] = ctx.beta2 * ctx.v[ i ] + ( 1.0 - ctx.beta2 ) * g * g

		mhat := ctx.m[ i ] / ctx.b1t
		vhat := ctx.v[ i ] / ctx.b2t
		ctx.param[ i ] -= ctx.lr * ( mhat / ( f32( math.sqrt( f64( vhat ) ) ) + ctx.eps ) )
	}
}

//
// Parallel API wrappers
//

oalinear_quantize_coef_par :: proc ( pool : ^Thread_Pool,
                                     L    : ^OA_Linear) {

	q := Quant_Ctx{

	    L = L
	}
	pool_run( pool, job_quantize, & q )
}

oalinear_forward_par :: proc ( pool  : ^Thread_Pool,
                               L     : ^OA_Linear,
                               x     : [ ]f32,
                               batch : int,
                               y     : [ ]f32,
                               pre   : [ ]f32 ) {

	oalinear_quantize_coef_par( pool, L )
	f := Fwd_Ctx{

	    L     = L,
	    x     = x,
		batch = batch,
	    y     = y,
		pre   = pre
	}

	pool_run( pool, job_forward, & f )
}

relu_forward_par :: proc ( pool : ^Thread_Pool,
                           x    : [ ]f32,
                           y    : [ ]f32 ) {

	r := Relu_Fwd_Ctx{

	    x = x,
		y = y
	}
	pool_run( pool, job_relu_fwd, & r )
}

relu_backward_par :: proc ( pool   : ^Thread_Pool,
                            y_relu : [ ]f32,
                            grad   : [ ]f32) {

	r := Relu_Bwd_Ctx{

	    y_relu = y_relu,
		grad   = grad
	}

	pool_run( pool, job_relu_bwd, & r )
}

softmax_xent_par :: proc ( pool    : ^Thread_Pool,
                           logits  : [ ]f32,
                           labels  : [ ]u8,
                           batch   : int,
                           dlogits : [ ]f32 ) ->
                           f32 {

	s : Softmax_Ctx
	s.logits  = logits
	s.labels  = labels
	s.batch   = batch
	s.dlogits = dlogits
	pool_run( pool, job_softmax_xent, & s )

	loss_sum : f32 = 0
	for t in 0 ..< NUM_THREADS {

		loss_sum += s.loss_part[ t ]
	}
	return loss_sum / f32( batch )
}

oalinear_zero_grads_par :: proc ( pool : ^Thread_Pool,
                                  L    : ^OA_Linear ) {

	z := Zero_Grad_Ctx{

	    L = L
	}
	pool_run( pool, job_zero_grads, & z )
}

oalinear_backward_par :: proc ( pool  : ^Thread_Pool,
                                L     : ^OA_Linear,
                                x     : [ ]f32,
                                pre   : [ ]f32,
                                dY    : [ ]f32,
                                batch : int,
                                dX    : [ ]f32 ) {

	oalinear_zero_grads_par( pool, L )

	bs := GBias_Scale_Ctx{

	    L     = L,
		pre   = pre,
	    dY    = dY,
		batch = batch
	}

	pool_run( pool, job_gbias_gscale, & bs )

	gc := GCoef_Ctx{

	    L     = L,
		x     = x,
	    dY    = dY,
		batch = batch
	}
	pool_run( pool, job_gcoef, & gc )

	dx := DX_Ctx{

	    L     = L,
		dY    = dY,
	    batch = batch,
		dX    = dX
	}

	pool_run( pool, job_dx, & dx )
}

oalinear_adam_step_par :: proc ( pool : ^Thread_Pool,
                                 L    : ^OA_Linear,
                                 lr   : f32,
                                 wd   : f32,
                                 t    : int ) {

	cs := coef_size( L )
	b1t := 1.0 - f32( math.pow( f64( ADAM_BETA1 ), f64( t ) ) )
	b2t := 1.0 - f32( math.pow( f64( ADAM_BETA2 ), f64( t ) ) )

	a := Adam_Ctx{

		param = L.coef,
	    grad  = L.g_coef,
		m     = L.m_coef,
	    v     = L.v_coef,
		n     = cs,
	    lr    = lr,
		beta1 = ADAM_BETA1,
	    beta2 = ADAM_BETA2,
		eps   = ADAM_EPS,
	    wd    = wd,
		b1t   = b1t,
	    b2t   = b2t,
	}

	pool_run( pool, job_adam, & a )

	// Scale and bias serial
	for o in 0 ..< L.out_dim {

		// Scale
		g := L.g_scale[ o ]
		L.m_scale[ o ] = ADAM_BETA1 * L.m_scale[ o ] + ( 1.0 - ADAM_BETA1 ) * g
		L.v_scale[ o ] = ADAM_BETA2 * L.v_scale[ o ] + ( 1.0 - ADAM_BETA2 ) * g * g
		mhat := L.m_scale[ o ] / b1t
		vhat := L.v_scale[ o ] / b2t
		L.scale[ o ] -= lr * ( mhat / ( f32( math.sqrt( f64( vhat ) ) ) + ADAM_EPS ) )

		// Bias
		g2 := L.g_bias[ o ]
		L.m_bias[ o ] = ADAM_BETA1 * L.m_bias[ o ] + ( 1.0 - ADAM_BETA1 ) * g2
		L.v_bias[ o ] = ADAM_BETA2 * L.v_bias[ o ] + ( 1.0 - ADAM_BETA2 ) * g2 * g2
		mhat2 := L.m_bias[ o ] / b1t
		vhat2 := L.v_bias[ o ] / b2t
		L.bias[ o ] -= lr * ( mhat2 / ( f32( math.sqrt( f64( vhat2 ) ) ) + ADAM_EPS ) )
	}
}

//
// Inference layer
//

OA_Linear_Inf :: struct {

	in_dim    : int,
    out_dim   : int,
	m         : int,
    N         : int,
    K         : int,
	blocks    : int,
	inv_sqrtK : f32,
	basis     : [ ]i8,
	scale     : [ ]f32,
	bias      : [ ]f32,
	cq        : [ ]i8,
}

inf_cq_size :: proc "contextless" (
                      L : ^OA_Linear_Inf ) ->
                      int {

	return L.out_dim * L.blocks * L.K
}

oalinear_inf_init :: proc ( L       : ^OA_Linear_Inf,
                            in_dim  : int,
                            out_dim : int,
                            B       : ^OA_Basis ) {

	mem.zero( L, size_of( L^ ) )
	L.in_dim  = in_dim
	L.out_dim = out_dim
	L.m = B.m
	L.N = B.N
	L.K = B.K
	L.blocks = ( in_dim + L.N - 1 ) / L.N
	L.inv_sqrtK = 1.0 / f32( math.sqrt( f64( L.K ) ) )
	L.basis = B.A
	L.scale = make( [ ]f32, out_dim )
	L.bias  = make( [ ]f32, out_dim )
	L.cq    = make( [ ]i8, inf_cq_size( L ) )
}

oalinear_inf_free :: proc ( L : ^OA_Linear_Inf ) {

	if len( L.scale ) > 0 {

	    delete( L.scale )
	}

	if len( L.bias ) > 0 {

	    delete( L.bias )
	}

	if len( L.cq ) > 0 {

	    delete( L.cq )
	}

	mem.zero( L, size_of( L^ ) )
}

oalinear_inf_forward :: proc "contextless" (
                               L     : ^OA_Linear_Inf,
                               x     : [ ]f32,
                               batch : int,
                               y     : [ ]f32 ) {

	for s in 0 ..< batch {

		xs := x[ s * L.in_dim : ( s + 1 ) * L.in_dim ]
		ys := y[ s * L.out_dim : ( s + 1 ) * L.out_dim ]

		for o in 0 ..< L.out_dim {

			acc : f32 = 0
			for i in 0 ..< L.in_dim {

			    block    := i / L.N
				n        := i - block * L.N
				base_row := n * L.K
				cq_off   := ( o * L.blocks + block ) * L.K

				dot := 0
				for k in 0 ..< L.K {

					a := int( L.basis[ base_row + k ])
					c := int( L.cq[ cq_off + k ])
					dot += a * c
				}
				acc += xs[ i ] * ( f32( dot ) * L.inv_sqrtK )
			}

			ys[ o ] = L.bias[ o ] + L.scale[ o ] * acc
		}
	}
}

relu_forward :: proc "contextless" (
                       x : [ ]f32,
                       y : [ ]f32 ) {

	for i in 0 ..< len( x ) {

		v := x[ i ]
		y[ i ] = ( v > 0 ) ? v : 0
	}
}

//
// Model save / load
//

save_layer_minimal :: proc ( fd   : os.Handle,
                             pool : ^Thread_Pool,
                             L_   : ^OA_Linear ) {

	L := L_
	oalinear_quantize_coef_par( pool, L )

	write_u32_le( fd, u32( L.in_dim ) )
	write_u32_le( fd, u32( L.out_dim ) )
	write_u32_le( fd, u32( L.blocks ) )
	write_u32_le( fd, u32( L.K ) )

	scale_q := make( [ ]i16, L.out_dim )
	bias_q  := make( [ ]i16, L.out_dim )
	defer {

	    delete( scale_q )
		delete( bias_q )
	    }

	scale_qstep : f32
	bias_qstep  : f32
	quantize_f32_to_i16( L.scale, & scale_qstep, scale_q )
	quantize_f32_to_i16( L.bias,  & bias_qstep,  bias_q )

	write_f32_le( fd, scale_qstep )
	for o in 0 ..< L.out_dim {

	    write_i16_le( fd, scale_q[ o ] )
	}

	write_f32_le( fd, bias_qstep )
	for o in 0 ..< L.out_dim {

	    write_i16_le( fd, bias_q[ o ] )
	}

	trit_count   := coef_size( L )
	packed_bytes := packed_bytes_for_trits( trit_count )

	write_u32_le( fd, u32( trit_count ) )
	write_u32_le( fd, u32( packed_bytes ) )

	buf := make( [ ]u8, packed_bytes )
	defer delete( buf )
	pack_trits_5perbyte( L.cq, buf )
	n, err := os.write( fd, buf )
	if err != os.ERROR_NONE || n != len( buf ) {

		fatal( "write packed trits failed" )
	}
}

load_layer_minimal :: proc ( fd : os.Handle,
                             L  : ^OA_Linear_Inf ) {

	in_dim  := int( read_u32_le( fd ) )
	out_dim := int( read_u32_le( fd ) )
	blocks  := int( read_u32_le( fd ) )
	K       := int( read_u32_le( fd ) )

	if in_dim != L.in_dim || out_dim != L.out_dim {

		fatal( "layer dim mismatch" )
	}
	if blocks != L.blocks || K != L.K {

		fatal( "layer blocks/K mismatch" )
	}

	scale_qstep := read_f32_le( fd )
	scale_q     := make( [ ]i16, L.out_dim )
	defer delete( scale_q )
	for o in 0 ..< L.out_dim {

	    scale_q[ o ] = read_i16_le( fd )
	}

	dequantize_i16_to_f32( scale_q, scale_qstep, L.scale )

	bias_qstep := read_f32_le( fd )
	bias_q := make( [ ]i16, L.out_dim )
	defer delete( bias_q )
	for o in 0 ..< L.out_dim {

	    bias_q[ o ] = read_i16_le( fd )
	}
	dequantize_i16_to_f32( bias_q, bias_qstep, L.bias )

	trit_count := int( read_u32_le( fd ) )
	packed_bytes := int( read_u32_le( fd ) )

	if trit_count != inf_cq_size( L ) {

		fatal( "trit_count mismatch" )
	}
	if packed_bytes != packed_bytes_for_trits( trit_count ) {

		fatal( "packed_bytes mismatch" )
	}

	buf := make( [ ]u8, packed_bytes )
	defer delete( buf )
	n, err := os.read_full( fd, buf )
	if err != os.ERROR_NONE || n != len( buf ) {

		fatal( "read packed trits failed" )
	}

	unpack_trits_5perbyte( buf, trit_count, L.cq )
}

save_model_minimal :: proc ( path   : string,
                             pool   : ^Thread_Pool,
                             oa_m   : int,
                             K_used : int,
                             L1     : ^OA_Linear,
                             L2     : ^OA_Linear,
                             L3     : ^OA_Linear ) {

	fd, err := os.open( path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o664 )
	if err != os.ERROR_NONE {

		fatal( "cannot open model for write: %s", path )
	}
	defer os.close( fd )

	_, errw := os.write_string( fd, MODEL_MAGIC )
	if errw != os.ERROR_NONE {

		fatal( "write magic failed" )
	}

	write_u32_le( fd, u32( MODEL_VERSION ) )
	write_u32_le( fd, u32( oa_m ) )
	write_u32_le( fd, u32( K_used ) )

	save_layer_minimal( fd, pool, L1 )
	save_layer_minimal( fd, pool, L2 )
	save_layer_minimal( fd, pool, L3 )
}

peek_model_header :: proc ( path       : string,
                            oa_m_out   : ^int,
                            K_used_out : ^int ) ->
                            bool {

	fd, err := os.open( path, os.O_RDONLY, 0 )
	if err != os.ERROR_NONE {

		return false
	}
	defer os.close( fd )

	magic : [ 4 ]u8
	n, er := os.read_full( fd, magic[ : ] )
	if er != os.ERROR_NONE || n != 4 {

		return false
	}
	if magic[ 0 ] != 'T' ||
       magic[ 1 ] != 'G' ||
       magic[ 2 ] != 'C' ||
       magic[ 3 ] != 'M' {

		return false
	}

	ver := int( read_u32_le( fd ) )
	if ver != MODEL_VERSION {

		fatal( "model version %d not supported ( expected %d )",
	           ver, MODEL_VERSION )
	}

	oa_m_out^   = int( read_u32_le( fd ) )
	K_used_out^ = int( read_u32_le( fd ) )

	return true
}

load_model_minimal :: proc ( path       : string,
                             oa_m_out   : ^int,
                             K_used_out : ^int,
                             L1         : ^OA_Linear_Inf,
                             L2         : ^OA_Linear_Inf,
                             L3         : ^OA_Linear_Inf ) ->
                             bool {

	fd, err := os.open( path, os.O_RDONLY, 0 )
	if err != os.ERROR_NONE {

		return false
	}
	defer os.close( fd )

	magic: [ 4 ]u8
	n, er := os.read_full( fd, magic[ : ] )
	if er != os.ERROR_NONE || n != 4 {

		return false
	}
	if magic[ 0 ] != 'T' ||
       magic[ 1 ] != 'G' ||
       magic[ 2 ] != 'C' ||
       magic[ 3 ] != 'M' {

		fatal( "magic mismatch" )
	}

	ver := int( read_u32_le( fd ) )
	if ver != MODEL_VERSION {

		fatal( "unsupported version" )
	}

	oa_m   := int( read_u32_le( fd ) )
	K_used := int( read_u32_le( fd ) )
	oa_m_out^   = oa_m
	K_used_out^ = K_used

	if L1.K != K_used ||
       L2.K != K_used ||
       L3.K != K_used {

		fatal( "K_used mismatch ( model %d, basis %d )", K_used, L1.K )
	}

	load_layer_minimal( fd, L1 )
	load_layer_minimal( fd, L2 )
	load_layer_minimal( fd, L3 )

	return true
}

//
// utils_2
//

build_mnist_paths :: proc ( folder        : string,
	                        train_img_buf : [ ]u8,
							train_lbl_buf : [ ]u8,
						    test_img_buf  : [ ]u8,
							test_lbl_buf  : [ ]u8 ) ->
                          ( train_img : string,
                            train_lbl : string,
                            test_img  : string,
                            test_lbl  : string ) {

	train_img = fmt.bprintf( train_img_buf, "%s/train-images-idx3-ubyte", folder )
	train_lbl = fmt.bprintf( train_lbl_buf, "%s/train-labels-idx1-ubyte", folder )
	test_img  = fmt.bprintf( test_img_buf,  "%s/t10k-images-idx3-ubyte", folder )
	test_lbl  = fmt.bprintf( test_lbl_buf,  "%s/t10k-labels-idx1-ubyte", folder )
	return
}

evaluate_accuracy_par :: proc ( pool  : ^Thread_Pool,
                                L1    : ^OA_Linear,
                                L2    : ^OA_Linear,
                                L3    : ^OA_Linear,
                                test  : ^MNIST,
                                batch : int ) ->
                                f32 {

	x := make( [ ]f32, batch * INPUT_DIM )
	defer delete( x )

	z1 := make( [ ]f32, batch * H1 )
	a1 := make( [ ]f32, batch * H1 )
    p1 := make( [ ]f32, batch * H1 )
	defer {

	    delete( z1 )
		delete( a1 )
	    delete( p1 )
	}

	z2 := make( [ ]f32, batch * H2 )
    a2 := make( [ ]f32, batch * H2 )
    p2 := make( [ ]f32, batch * H2 )
	defer {

	    delete( z2 )
		delete( a2 )
		delete( p2 )
    }

	logits := make( [ ]f32, batch * OUT )
    p3 := make( [ ]f32, batch * OUT )
	defer {

	    delete( logits )
		delete( p3 )
	}

	correct := 0
	total   := 0

	for start := 0; start < test.count; start += batch {

		bsz := batch
		if start + bsz > test.count {

		    bsz = test.count - start
		}
		if bsz <= 0 {

		    continue
	    }

		for s := 0; s < bsz; s += 1 {

		    idx := start + s
			img := test.images[ idx * INPUT_DIM : ( idx + 1 ) * INPUT_DIM ]
			xs := x[ s * INPUT_DIM : ( s + 1 ) * INPUT_DIM ]
			for i in 0 ..< INPUT_DIM {

				xs[ i ] = to_float_pixel( img[ i ] )
			}
		}

		oalinear_forward_par( pool, L1, x, bsz, z1, p1 )
		relu_forward_par( pool, z1[ 0 : bsz * H1 ], a1[ 0 : bsz * H1 ] )

		oalinear_forward_par( pool, L2, a1, bsz, z2, p2 )
		relu_forward_par( pool, z2[ 0 : bsz * H2 ], a2[ 0 : bsz * H2 ] )

		oalinear_forward_par( pool, L3, a2, bsz, logits, p3 )

		for s in 0 ..< bsz {

			ls := logits[ s * OUT : ( s + 1 ) * OUT ]
			pred := 0
			best := ls[ 0 ]
			for j := 1; j < OUT; j += 1 {

				if ls[ j ] > best {

					best = ls[ j ]
					pred = j
				}
			}
			truth := int( test.labels[ start + s ] )
			if pred == truth {

			    correct += 1
		    }
			total += 1
		}
	}

	if total <= 0 {

	    return 0
	}

	return f32( correct ) / f32( total )
}

lr_log_schedule :: proc ( step        : int,
                          total_steps : int,
                          lr_start    : f32,
                          lr_end      : f32 ) ->
                          f32 {

	if total_steps <= 1 {

		return lr_end
	}
	s := step
	if s < 1 {

	    s = 1
	}
	if s > total_steps {

	    s = total_steps
	}

	t := f32( s - 1 ) / f32( total_steps - 1 )
	a := f32( math.ln( f64( lr_start ) ) )
	b := f32( math.ln( f64( lr_end ) ) )
	return f32( math.exp( f64( a + t * ( b - a ) ) ) )
}

usage :: proc ( prog : string ) {

	fmt.printf( "Usage:\n" )
	fmt.printf( "  %s train <mnist_folder> <model.bin> [epochs]\n", prog )
	fmt.printf( "  %s infer <mnist_folder> <model.bin> <train|test> <index>\n", prog )
}

//
// main
//

main :: proc() {

    // seed RNG
	seed := u64( time.time_to_unix( time.now( ) ) )
	rand.reset_u64( seed )

	args := os.args
	if len( args ) < 2 {

		usage( args[ 0 ] )
		return
	}

	if args[ 1 ] == "train" {

		if len( args ) < 4 {

			usage( args[ 0 ] )
			return
		}
		mnist_folder := args[ 2 ]
		model_path   := args[ 3 ]
		epochs       := DEFAULT_EPOCHS
		if len( args ) >= 5 {

		    epochs, _ = strconv.parse_int( args[ 4 ], 10 )
			if epochs <= 0 {

			    epochs = DEFAULT_EPOCHS
		    }
		}

		// Init thread pool
		pool : Thread_Pool
		pool_init( & pool )
		defer pool_destroy( & pool )

		// Build MNIST paths using stack buffers
		train_img_buf : [ 512 ]u8
		train_lbl_buf : [ 512 ]u8
		test_img_buf  : [ 512 ]u8
		test_lbl_buf  : [ 512 ]u8
		train_img, train_lbl, test_img, test_lbl :=
			build_mnist_paths( mnist_folder,
				               train_img_buf[ : ],
							   train_lbl_buf[ : ],
							   test_img_buf[ : ],
							   test_lbl_buf[ : ] )

		fmt.println( "Loading MNIST..." )
		train := load_mnist( train_img, train_lbl )
		test  := load_mnist( test_img, test_lbl )
		defer {

		    free_mnist( & train )
			free_mnist( & test )
	    }

		fmt.printfln( "Train: %d  Test: %d", train.count, test.count )

		oa_m   := DEFAULT_OA_M
		K_used := K_USED_DEFAULT

		full := build_oa_basis_full( oa_m )
		defer free_oa_basis( & full )
		if K_used > full.K {

			fatal( "K_used=%d > K_full=%d", K_used, full.K )
		}
		B := trim_basis( & full, K_used )
		defer free_oa_basis( & B )

		fmt.printfln( "OA basis: m=%d N=%d K_used=%d", B.m, B.N, B.K )
		fmt.printfln( "Threads: %d", NUM_THREADS )

		L1 : OA_Linear
	    L2 : OA_Linear
		L3 : OA_Linear
		oalinear_init( & L1, INPUT_DIM, H1, & B, TERNARY_ALPHA )
		oalinear_init( & L2, H1, H2, & B, TERNARY_ALPHA )
		oalinear_init( & L3, H2, OUT, & B, TERNARY_ALPHA )
		defer {

		    oalinear_free( & L1 )
			oalinear_free( & L2 )
		    oalinear_free( & L3 )
	    }

		batch := DEFAULT_BATCH

		x   := make( [ ]f32, batch * INPUT_DIM )
		z1  := make( [ ]f32, batch * H1 )
		a1  := make( [ ]f32, batch * H1 )
		p1  := make( [ ]f32, batch * H1 )
		dz1 := make( [ ]f32, batch * H1 )

		z2  := make( [ ]f32, batch * H2 )
		a2  := make( [ ]f32, batch * H2 )
		p2  := make( [ ]f32, batch * H2 )
		dz2 := make( [ ]f32, batch * H2 )

		logits  := make( [ ]f32, batch * OUT )
		p3      := make( [ ]f32, batch * OUT )
		dlogits := make( [ ]f32, batch * OUT )

		da2 := make( [ ]f32, batch * H2 )
		da1 := make( [ ]f32, batch * H1 )
		dx0 := make( [ ]f32, batch * INPUT_DIM )

		defer {

			delete( x)
			delete( z1 )
		    delete( a1 )
			delete( p1 )
		    delete( dz1 )
			delete( z2 )
		    delete( a2 )
			delete( p2 )
		    delete( dz2 )
			delete( logits )
		    delete( p3 )
			delete( dlogits )
			delete( da2 )
		    delete( da1 )
			delete( dx0 )
		}

		perm := make( [ ]int, train.count )
		defer delete( perm )
		for i in 0 ..< train.count {

		    perm[ i ] = i
	    }

		step := 0
		total_steps := ( ( train.count + batch - 1 ) / batch ) * epochs

		for ep := 1; ep <= epochs; ep += 1 {

			// Shuffle
			for i := train.count - 1; i > 0; i -= 1 {

				j := rand.int_range( 0, i + 1 )
				perm[ i ], perm[ j ] = perm[ j ], perm[ i ]
			}

			epoch_loss_sum : f32 = 0
			seen := 0

			for start := 0; start < train.count; start += batch {

				bsz := batch
				if start + bsz > train.count {

				    bsz = train.count - start
			    }
				if bsz <= 0 {

				    continue
			    }

				yb := make( [ ]u8, bsz )
				defer delete( yb )

				for s := 0; s < bsz; s += 1 {

					idx    := perm[ start + s ]
					yb[ s ] = train.labels[ idx ]
					img    := train.images[ idx * INPUT_DIM : ( idx + 1 ) * INPUT_DIM ]
					xs     := x[ s * INPUT_DIM : ( s + 1 ) * INPUT_DIM ]
					for i := 0; i < INPUT_DIM; i += 1 {

						xs[ i ] = to_float_pixel( img[ i ] )
					}
				}

				// Forward ( parallel )
				oalinear_forward_par( & pool, & L1, x, bsz, z1, p1 )
				relu_forward_par( & pool, z1[ 0 : bsz * H1 ], a1[ 0 : bsz * H1 ] )

				oalinear_forward_par( & pool, & L2, a1, bsz, z2, p2 )
				relu_forward_par( & pool, z2[ 0 : bsz * H2 ], a2[ 0 : bsz * H2 ] )

				oalinear_forward_par( & pool, & L3, a2, bsz, logits, p3 )

				// Loss + dlogits ( parallel )
				loss := softmax_xent_par( & pool, logits[ 0 : bsz * OUT ], yb, bsz, dlogits[ 0 : bsz * OUT ] )
				epoch_loss_sum += loss * f32( bsz )
				seen += bsz

				// Backward ( parallel )
				oalinear_backward_par( & pool, & L3, a2, p3, dlogits, bsz, da2 )
				relu_backward_par( & pool, a2[ 0 : bsz * H2 ], da2[ 0 : bsz * H2 ] )
				copy( dz2[ 0 : bsz * H2 ], da2[ 0 : bsz * H2 ] )

				oalinear_backward_par( & pool, & L2, a1, p2, dz2, bsz, da1 )
				relu_backward_par( & pool, a1[ 0 : bsz * H1 ], da1[ 0 : bsz * H1 ] )
				copy( dz1[ 0 : bsz * H1 ], da1[ 0 : bsz * H1 ] )

				oalinear_backward_par( & pool, & L1, x, p1, dz1, bsz, dx0 )

				// Adam step with log schedule
				step += 1
				lr_now := lr_log_schedule( step, total_steps, LR_START, LR_END )

				oalinear_adam_step_par( & pool, & L1, lr_now, WEIGHT_DECAY, step )
				oalinear_adam_step_par( & pool, & L2, lr_now, WEIGHT_DECAY, step )
				oalinear_adam_step_par( & pool, & L3, lr_now, WEIGHT_DECAY, step )
			}

			avg_loss := epoch_loss_sum / f32( seen )
			test_acc := evaluate_accuracy_par( & pool, & L1, & L2, & L3, & test, batch )
			lr_now := lr_log_schedule( step, total_steps, LR_START, LR_END )

			fmt.printfln( "Epoch %d/%d | loss %.4f | test acc %.2f%% | lr %.6g",
				          ep, epochs, avg_loss, 100.0*test_acc, lr_now )
		}

		fmt.printfln( "Saving minimal model to %s ...", model_path )
		save_model_minimal( model_path, & pool, oa_m, K_used, & L1, & L2, & L3 )
		fmt.println( "Saved." )
		return
	}

	if args[ 1 ] == "infer" {

		if len( args ) < 6 {

			usage( args[ 0 ] )
			return
		}
		mnist_folder := args[ 2 ]
		model_path   := args[ 3 ]
		split        := args[ 4 ]

		// fmt.printfln( "%v", args[ 5 ] )

        index, _ := strconv.parse_int( args[ 5 ], 10 )

		train_img_buf: [ 512 ]u8
		train_lbl_buf: [ 512 ]u8
		test_img_buf:  [ 512 ]u8
		test_lbl_buf:  [ 512 ]u8
		train_img, train_lbl, test_img, test_lbl :=
			build_mnist_paths( mnist_folder,
				               train_img_buf[ : ],
							   train_lbl_buf[ : ],
							   test_img_buf[ : ],
							   test_lbl_buf[ : ] )

		train : MNIST
	    test  : MNIST
		ds : ^MNIST = nil
		if split == "train" {

			train = load_mnist( train_img, train_lbl )
			// defer free_mnist(&train)
			ds = & train
		} else if split == "test" {

			test = load_mnist( test_img, test_lbl )
			// defer free_mnist(&test)
			ds = & test
		} else {

			fatal( "split must be train or test" )
		}

		if index < 0 || index >= ds.count {

			fatal( "index out of range" )
		}

		oa_m   : int
		K_used : int
		if !peek_model_header( model_path, & oa_m, & K_used ) {

			fatal( "cannot read model header" )
		}

		full := build_oa_basis_full( oa_m )
		defer free_oa_basis( & full )
		if K_used > full.K {

			fatal( "model K_used > K_full" )
		}
		B := trim_basis( & full, K_used )
		defer free_oa_basis( & B )

		L1 : OA_Linear_Inf
	    L2 : OA_Linear_Inf
		L3 : OA_Linear_Inf
		oalinear_inf_init( & L1, INPUT_DIM, H1, & B )
		oalinear_inf_init( & L2, H1, H2, & B )
		oalinear_inf_init( & L3, H2, OUT, & B )
		defer {

		    oalinear_inf_free( & L1 )
			oalinear_inf_free( & L2 )
			oalinear_inf_free( & L3 )
	    }

		oa_m2   : int
		K_used2 : int
		if !load_model_minimal( model_path, & oa_m2, & K_used2, & L1, & L2, & L3 ) {

			fatal( "load_model failed" )
		}

		x : [ INPUT_DIM ]f32
		img := ds.images[ index * INPUT_DIM : ( index + 1 ) * INPUT_DIM ]
		for i := 0; i < INPUT_DIM; i += 1 {

			x[ i ] = to_float_pixel( img[ i ] )
		}

		z1     : [ H1 ]f32
		a1     : [ H1 ]f32
		z2     : [ H2 ]f32
		a2     : [ H2 ]f32
		logits : [ OUT ]f32

		oalinear_inf_forward( & L1, x[ : ], 1, z1[ : ] )
		relu_forward( z1[ : ], a1[ : ] )
		oalinear_inf_forward( & L2, a1[ : ], 1, z2[ : ] )
		relu_forward( z2[ : ], a2[ : ] )
		oalinear_inf_forward( & L3, a2[ : ], 1, logits[ : ] )

		pred := 0
		best := logits[ 0 ]
		for j := 1; j < OUT; j += 1 {

			if logits[ j ] > best {

				best = logits[ j ]
				pred = j
			}
		}
		truth := int( ds.labels[ index ] )

		fmt.printfln( "Split: %s | index: %d | predicted: %d | true: %d | (oa_m=%d K_used=%d)",
			split, index, pred, truth, oa_m2, K_used2 )
		return
	}

	usage( args[ 0 ] )
}
