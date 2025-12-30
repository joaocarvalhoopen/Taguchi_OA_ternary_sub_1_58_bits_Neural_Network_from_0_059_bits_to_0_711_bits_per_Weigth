# Taguchi OA ternary sub 1.58 bits Neural Network from 0.059 bits to 0.711 bits per Weigth
High compression Weight encoded training with Orthogonal Arrays, Taguchi of ternary -1, 0, 1 weights of 1.58 bits style Neural Network.

## Description
- OA - Orthogonal Arrays / Taguchi ternary coefficient layers + STE-style training
- Minimal model save : packed trits ( 5 trits per byte ) + int16 scale / bias
- Parallel training : 24 threads via pthreads

## MNIST folder must contain uncompressed IDX files:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte

## How to compile and run

```
Build:
    odin build . -out:taguchi_ternary_mnist_neural_network.exe -o:speed -no-bounds-check

Run:
    ./taguchi_ternary_mnist_neural_network.exe train <mnist_folder> <model.bin> [epochs]
    ./taguchi_ternary_mnist_neural_network.exe <mnist_folder> <model.bin> <train|test> <index>
```

### How to calculate the used bits in the weights

* example 0.533 is the ideal ( without padding )

```
  N = 3^m
  bits/weight = ideal = 1.6 * K / N
  
  In here we use the aproximated 1.6 ~ 1.58,
  from the ln( 3 ) for the trits { -1 , 0, 1 }
  
  Example:
  
  for m = 2 -> N = 9 and K = 3 : ( 1.6  x  3 / 9  = 0.5333 )
```

* But in pratice we have `ceil( in_dim / N )` blocks ( There is padding until the multiple N) and the file saves each layer  separatelly with  `ceil( trits / 5 )` that is 5 trits in each byte, there is same rounding in here. This makes the value  a little bit higher 0.539 bits/weight  ( In disk there are only trits ) and a little bit more if you count file header, scale and bias.

Bellow is a table with all the calculations for many m and K.

* Current neural neteorks arquitectures, fully connected layers `784 -> 400 -> 128 -> 10` with a softmax at the end.

Description of the columns in the following tables:

* bits/weight ideal = ( 1.6 x K / N ) Teorica reference. 
* bits/weight trits( disk ) = Only ternary packed coeficientes packed ( 5 trits / byte ).
* bits/weight file( total ) = Complete file ( model.bin ) with headers + scale / bias quantized.
* KiB = Final size of the file.
* x smaller vs int8 / fp16 / fp32 = Comparation with saving only dense wights in those formats.


## Current Neural Networkd 784 -> 400 -> 128 -> 10

* Dense Weights : 366_080

* Size (only weights) if they were : int8 = 366_080 B ( 357.50 KiB ), fp16 = 732_160 B ( 715.00 KiB ), fp32 = 1_464_320 B ( 1430.00 KiB ).

### m = 2, N = 9, K_max = 4

```
|  K | bits_w_ideal | bits_w_trits | bits_w_file | KiB   | x_int8 | x_fp16 | x_fp32  |
--------------------------------------------------------------------------------------
|  1 | 0.178        | 0.180        | 0.229       | 10.24 | 34.91× | 69.82× | 139.65× |     <----- Epoch=300 loss_0.9797 test_acc  70.05 %
|  2 | 0.356        | 0.359        | 0.409       | 18.27 | 19.57× | 39.14× | 78.27×  |
|  3 | 0.533        | 0.539        | 0.589       | 26.30 | 13.59× | 27.19× | 54.38×  |     <---- Good Test_Accuracy epoch 236  96.41 %
|  4 | 0.711        | 0.719        | 0.768       | 34.33 | 10.41× | 20.83× | 41.66×  |

```

### m = 3, N = 27, K_max = 13

```
|  K | bits_w_ideal | bits_w_trits | bits_w_file | KiB   | x_int8 | x_fp16  | x_fp32  |
---------------------------------------------------------------------------------------
|  1 | 0.059        | 0.060        | 0.109       | 4.99  | 71.70× | 143.40× | 286.80× |
|  2 | 0.119        | 0.121        | 0.170       | 7.76  | 46.14× | 92.29×  | 184.57× |
|  3 | 0.178        | 0.181        | 0.230       | 10.54 | 33.97× | 67.94×  | 135.88× |
|  4 | 0.237        | 0.242        | 0.291       | 13.31 | 26.89× | 53.79×  | 107.58× |
|  5 | 0.296        | 0.302        | 0.351       | 16.08 | 22.24× | 44.48×  | 88.96×  |
|  6 | 0.356        | 0.363        | 0.412       | 18.85 | 18.97× | 37.95×  | 75.90×  |
|  7 | 0.415        | 0.423        | 0.472       | 21.62 | 16.57× | 33.14×  | 66.29×  |
|  8 | 0.474        | 0.484        | 0.533       | 24.40 | 14.73× | 29.47×  | 58.93×  |
|  9 | 0.533        | 0.544        | 0.594       | 27.17 | 13.27× | 26.53×  | 53.07×  |
| 10 | 0.593        | 0.605        | 0.654       | 29.94 | 12.08× | 24.15×  | 48.30×  |
| 11 | 0.652        | 0.665        | 0.715       | 32.71 | 11.09× | 22.18×  | 44.36×  |
| 12 | 0.711        | 0.726        | 0.775       | 35.48 | 10.25× | 20.50×  | 41.00×  |
| 13 | 0.770        | 0.786        | 0.836       | 38.25 | 9.52×  | 19.04×  | 38.08×  |

```

## References

1. Video Taguchi Arrays <br>
   [https://youtu.be/5oULEuOoRd0](https://youtu.be/5oULEuOoRd0)

2. Wikipedia Orthogonal array <br>
   [http://en.wikipedia.org/wiki/Orthogonal_array](http://en.wikipedia.org/wiki/Orthogonal_array)

3. Matlab Taguchi Array <br>
   [https://www.mathworks.com/matlabcentral/fileexchange/71628-taguchiarray?s_tid=mwa_osa_a](https://www.mathworks.com/matlabcentral/fileexchange/71628-taguchiarray?s_tid=mwa_osa_a)

4. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits <br>
   by Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei <br>
   [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)
  
5. BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks <br>
   by Jacob Nielsen, Peter Schneider-Kamp <br>
   [https://arxiv.org/abs/2407.09527](https://arxiv.org/abs/2407.09527)
  
6. GitHub Microsoft project bitnet.cpp <br>
   [https://github.com/microsoft/bitnet](https://github.com/microsoft/bitnet)

## License
MIT Open Source License

## Have fun
Best Regards, <br>
Joao Carvalho
