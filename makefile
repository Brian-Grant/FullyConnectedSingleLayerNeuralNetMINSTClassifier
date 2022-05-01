BUILDID=$(shell date +%m/%d/%Y-%H:%M:%S)

# GPUD			to define a cuda error checking macro
# PBATCH		to print batch loss
# NRAND			to seed rands with constand for debugging 

FLAGS += -DGPUD


LEARNING_RATE_1E-3 = 0.001
LEARNING_RATE_1E-4 = 0.0001
EPOCHS = 10
BATCH_SIZE = 100



# q = quiet mode
# r = random seed mode
# o = optimized mode

# compile

# verbose / constant seed / not optimized
all: mnist-nn 

# verbose / constant seed / optimized
o: mnist-nn-o

# verbose / random seed / not optimized 
r: mnist-nn-r

# verbose / random seed / optimized
or: mnist-nn-or

# quiet / constant seed / not optimized
q: mnist-nn-q

# quiet / constant seed / optimized
qo: mnist-nn-qo

# quiet / random seed / not optimised
qr: mnist-nn-qr

# quiet / random seed / optimized
qor: mnist-nn-qor


seq: stest sotest


# NN

mnist-nn: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DPBATCH -DNRAND -G -g -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-o: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DPBATCH -DNRAND -O3 -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-r: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DPBATCH -G -g -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-or: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DPBATCH -O3 -o mnist-nn driver.cu cuda_fns.cu



mnist-nn-q: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DNRAND -G -g -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-qo: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -DNRAND -O3 -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-qr: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -G -g -o mnist-nn driver.cu cuda_fns.cu

mnist-nn-qor: driver.cu mnist_parser.hpp cuda_fns.cu cuda_fns.hpp
	nvcc $(FLAGS) -O3 -o mnist-nn driver.cu cuda_fns.cu


run1: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-3) 1 $(BATCH_SIZE)

run2: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-3) 2 $(BATCH_SIZE)

run5: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-3) 5 $(BATCH_SIZE)

run10: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-3) 10 $(BATCH_SIZE)

lrun1: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-4) 1 $(BATCH_SIZE)

lrun2: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-4) 2 $(BATCH_SIZE)

lrun5: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-4) 5 $(BATCH_SIZE)

lrun10: mnist-nn
	./mnist-nn $(LEARNING_RATE_1E-4) 10 $(BATCH_SIZE)



gdb: mnist-nn
	gdb mnist-nn $(LEARNING_RATE_1E-3) 1 $(BATCH_SIZE)

mem: mnist
	cuda-memcheck ./mnist-nn $(LEARNING_RATE_1E-3) 1 $(BATCH_SIZE)

race: mnist
	cuda-memcheck --tool racecheck ./mnist-nn $(LEARNING_RATE_1E-3) 1 $(BATCH_SIZE)


# sequential -------------------------------------------------------------------
stest: sequential2.cpp
	g++ -std=c++1z -g -Wall -Wextra -pedantic -o sequential2 sequential2.cpp

srun: clear stest
	./sequential2

sclang: sequential2.cpp 
	clang++ -std=c++1z -g -Wall -Wextra -pedantic -o sequential2 sequential2.cpp

sval: sequential2.cpp clear
	valgrind -v --track-origins=yes --leak-check=full \
			--show-leak-kinds=all ./sequential2
sgdb: clear
	gdb sequential2



# softmax -------------------------------------------------------------------
sotest: softmax_cross-entropy.cpp
	g++ -std=c++1z -g -Wall -Wextra -pedantic -o softmax_cross-entropy softmax_cross-entropy.cpp

sorun: clear sotest
	./softmax_cross-entropy

soclang: softmax_cross-entropy.cpp 
	clang++ -std=c++1z -g -Wall -Wextra -pedantic -o softmax_cross-entropy softmax_cross-entropy.cpp

soval: softmax_cross-entropy.cpp clear
	valgrind -v --track-origins=yes --leak-check=full \
			--show-leak-kinds=all ./softmax_cross-entropy
sogdb: clear
	gdb softmax_cross-entropy


node:
	salloc -p gpu

sq:
	squeue -u bgrant11

mod:
	module load gnu8


git: cleanall
	git add -A .
	git commit -m "commit on $(BUILDID)"
	git push 


pull: cleanall
	git pull


clean:
	rm -f *.o sequential2 softmax_cross-entropy mnist-nn

cleanascii:
	rm -f training_ascii.txt test_ascii.txt

cleanall: cleanascii clean

clear:
	clear && clear
	




