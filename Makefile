all:
	odin build . -out:taguchi_mnist_odin.exe -o:speed

opti:
	odin build . -out:taguchi_mnist_odin.exe -o:speed -no-bounds-check -microarch=native

clean:
	rm -f ./taguchi_mnist_odin.exe

run_train:
	# ./taguchi_mnist_odin.exe train mnist model.bin 2
	./taguchi_mnist_odin.exe train mnist model.bin 300

run_infer:
	./taguchi_mnist_odin.exe infer mnist model.bin test 222
