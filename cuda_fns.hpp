#ifndef CUDA_FNS_HPP
#define CUDA_FNS_HPP


#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <fstream>
#include <numeric>
#include <vector>
#include <random>
#include <algorithm>
#include <stdio.h>
#include <limits>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>



// build model
// -----------------------------------------------------------------------------
__global__ void init_weights(float *W, int m, int n, uint64_t seed);



// forward 
// -----------------------------------------------------------------------------
__global__ void multiplyX1W1(float *X, float *W, float *Z, int m, int n); //XXX fn_id = 2

__global__ void reLu(float *z1, float *x2, float* relud); // no dropout. for test run
	
__global__ void reLu_train(float *z1, float *x2, float* relud, uint64_t seed); // has dropout with 0.4 chance

__global__ void multiplyX2W2(float *X, float *W, float *Z, int m, int n);

__global__ void softmax(float* z2, float * y_soft_predict);

// backward
// -----------------------------------------------------------------------------
__global__ void z2d_and_ce(float* y_soft_predict, unsigned char* y_true, float* z2d,
									float* epoch_ce, float* batch_ce, 
									int instance_idx, int batch_size, 
									int batch_idx, int num_batches);


__global__ void calcW2d(float *X2, float *z2d, float *w2d, int m_w2d, int n_w2d);

__global__ void calcz1d(float* z2d, float* W2, float* z1d, float* relud, int m_w2, int n_w2);

__global__ void calcW1d(float *X1, float *z1d, float *w1d, int m_w1d, int n_w1d);

__global__ void updateW1(float* w1, float* w1d, long batch_size, float learning_rate);

__global__ void updateW2(float* w2, float* w2d, long batch_size, float learning_rate);



// for test set run
// -----------------------------------------------------------------------------
__global__ void test_run(float* y_soft_predict, unsigned char * y_true, int * num_correct);





// kernel functions that are no longer in use
// -----------------------------------------------------------------------------

//__global__ void reduce(float *Zxw, float *Z, int m_Zxw, int n_Zxw);


//__global__ void reduceZ2(float *Zxw, float *Z, int m_Zxw, int n_Zxw);


//__global__ void multiplyX1W1(float *X, float *W, float *Zxw, int m, int n); //XXX fn_id = 1

//__global__ void reduceZ1(float *Zxw, float *Z, int m_Zxw, int n_Zxw);			//XXX fn_id = 1



// model verification 
// -----------------------------------------------------------------------------
__global__ void print_w(float* W, int m, int n);

__global__ void mulcheck(float *X, float *W, float *Zxw, int m, int n);

__global__ void xicheck(float* X, int len);

__global__ void reducecheck(float *Zxw, float *Z, int m_Zxw, int n_Zxw);

__global__ void reducemulcheck(float *X, float *W, float *Z, int m, int n);

__global__ void transposemulcheck(float *X2, float *z2d, float *w2d, float *w2d_accum, int m_w2d, int n_w2d);

__global__ void x2dreducecheck(float* z2d, float* W2, float* x2dz2dw2, float* x2d, int m_x2dz2dw2, int n_x2dz2dw2);



// sandbox experiments
// -----------------------------------------------------------------------------
__global__ void test();

__global__ void indexing_test();



#endif
