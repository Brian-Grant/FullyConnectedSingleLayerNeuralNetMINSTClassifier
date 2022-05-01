# 547_04

Fully connected neural network mnist classifier

run code with:

./mnist-nn <learning_rate> <epochs> <batch_size>

The makefile contains many different options for easy demos

The tldr for a quick demo is:

make or
make run10

The explanation is as follows:

There are 3 macros that affect compilation

GPUD			to define a cuda error checking macro
PBATCH			to print batch loss
NRAND			to seed rands with constand for debugging 

GPUD should always be defined. The only reason to have it not defined
is to collapse the cuda error checking macro to run the host code
on a machine that does not have a gpu

GPUD defines easy to use cuda macros which are located in nn_macros.hpp

PBATCH will make the program output batch loss at the completion of every batch
NRAND should be absent for a demo run to enable random seed

this is made easier with the following makefile rules

o - optimize
r - random seed
q - quiet mode (dont print batch loss)

verbose / constant seed / not optimized
    make 

verbose / constant seed / optimized
    make o

verbose / random seed / not optimized 
    make r

verbose / random seed / optimized
    make or

quiet / constant seed / not optimized
    make q

quiet / constant seed / optimized
    make qo

quiet / random seed / not optimised
    make qr

quiet / random seed / optimized
    make qor


Execution:
rules with no l have learning rate of 0.001
rules with l have learning rate of 0.0001
all have batch size of 100
number on rhs indicates number of epochs

make run1  
make run2
make run5
make run10
make lrun1
make lrun2
make lrun5
make lrun10

comments in driver.cu describe the general structure of the code
comments in cuda_fns.cu are more detailed descriptions of each kernel

In general, the code structure is as follows

Parse input files
malloc memory for model train
initialize weight matricies
set up random train set permutation
memcpy a batch of instances and labels
for each instance in the batch:
perform forward pass
multiply input against hidden neurons weights
feed through relu
multiply relu output with output layer weights
softmax
cross entropy
perform backprop
accumulate gradients for the weight matricies
upon batch completion
average weight gradients
update weights
get new batch, repeat

test
allocate device memory for whole test set
perform forward pass on each train set instance
tally number of correct predictions
end



This was a lot of fun to code, and I learned a lot about writing cuda code.
After impelenting something that works, I was able to incrementally optimize the 
cuda code. And with each optimization I was able to significantly push
down execution time while maintaing prediction accuracy

As of now the code can perform 10 epochs in just over 60 seconds
and obtain an accuracy of roughly 94% consistently

The 94% accuracy is achieved with 
10 epochs
100 batch size
0.001 learning rate

If you take a look at the code, and the comment descriptions, the 
block structure, and kernel code structure is all chosen for speed
I was able to implement 
loop unrolls
shared arrays, which are private to blocks where chosen to perform
    sections of calculations separately from other sections
    these sections are then combined
dot products are first multiplied, then added through a parallel reduction
I tried to implement a warp reduce but suspect that this portion of the code
returned nondeterministic results when compiled with optimizations 
and so they were removed

Between an optimized and non optimized compile there is a very slight difference
in results, even with a constant random seed.
I tried to remedy this with a bunch of extra __synchthreads() but the
very minor difference remains, but it is small enough to where I do not think
it is an issue leaving it in.

This program went through many many revisions as I understood the implications
of block structure, shared memory, coalesced memory access, loop unrolls, etc

Some of my earlier code were more general kernel functions which had more
if statements to handle the different sizes of the input arrays.

But I came to understand the performance degredation of if statemetns in 
cuda kernels, and replaced these more general kernels with kernels written 
for specific purposes, with expected input size, etc

In the case of reductions, I would handle the rightmost end of say, the
input array of length 785, first, to get the input size down to a power of 2
to allow for a more efficent reduction.

Some kernel functions started out as separate entities such as one for multiply
then a separate for the reduce, but as my understanding increased, I was able to 
combine multiply, then reduce in a single kernel function.

I structured the code in such a way to minimize memory transfers between
host and device.

The most frequent memory transfer is the memcpy of a batch of instances
at a time during model training.

This was because I wanted the code to be general enough to have a batch size
equal to the whole training set, and I was uncertain if a transfer that large
would cause problems. I wanted the code to be computationally efficient,
but memory efficiency was also taken into consideration. 

There are commented out timer macros in the train code, which I used to 
benchmark performance.

One timer macro is left uncommented, which outputs model train duration.
I was able to significantly reduce this duration with the optimizations implemented
along the way.

Near the beginning of driver.cu there are two commented out calls to the function
print_set()
If this is uncommented, the function will go through the train, and test set, and
print ascii representaions of the pixels in each 28 x 28 digit image.
It is just a neat way to visually see representations of the input to the
neural network.

This assignment was a ton of fun, and I absolutely learned a ton about cuda
programming.

In cuda_fns.cu after the kernels related to the model, there are a handful
of older versions of codes that were later optimized, but this only includes
a handful of changes made to make the code more efficiently parallel.

After the defunct kernel functions, there are a handful of model verification
functions I used to make sure the optimizations I was implementing were
consistent with less parallel/less efficient versions of the code.
These were extremely helpful forms of correctness verification

After the verification functionsthere are a few sandbox-esque test kernels I used
to experiment with block, and thread indexing. 

Thank you for your time. I thoroughly enjoyed this class, and hope to take my 
career in a direction that utilizes what I learned here.
