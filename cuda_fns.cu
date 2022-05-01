#include "cuda_fns.hpp"
#include "nn_macros.hpp"


// init weights with random values [-0.01, 0.01]
// seed with time
// or seed with constant for debugging
__global__ void init_weights(float *W, int m, int n, uint64_t seed){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int num_weights = m * n;	
	int init_ctr = 0;
	float rand_num;
	for(int idx = tid; idx < num_weights; idx += stride){	
		curandState randinfo;
		#ifdef NRAND
		curand_init(1 + tid + init_ctr, 0, 0, &randinfo); // constant seed for debugging	
		#else	
		curand_init(seed + tid + init_ctr, 0, 0, &randinfo);
		#endif		
		rand_num = curand_uniform(&randinfo);
		rand_num -= 0.5; // shift from [0. 1] -> [-0.5, 0.5]
		rand_num /= 50.0; // shrink from [-0.5, 0.5] -> [-0.01, 0.01]
		W[idx] = rand_num;
		init_ctr++;
	}
}



// multiplies one instance X with weight matrix W
// W contains all weights associated with the transition
//		input_layer -> hidden layer
// instanceZxw is a partially reduced product of X * Wi
// Wi is one weight vector of length 785, which is associated with
// one of the 1024 hidden neurons
// instanceZxw is then reduced via unrolled loop 
// Z is the vector of length 1024 associated with the dot product of 
// X and each of the 1024 Wis
// X is of len 785
// W is of len 1024 * 785
// this kernel is called with <<<1024, 256>>>
// thus each block gets a privately shared array associated with one
// of the 1024 hidden neurons 
// the block size is 256 and instanceZxw is partially reduced because it is
// more straight forward to reduce an array with a size of a power of 2

// as a side note when instanceZxw was of size 64, and thus only 32 threads
// are needed for the next stage of the reduction, I tried to implement a 
// warp reduce where syncthreads would not be needed but for an unknown reason
// the code would not result in deterministic results when compiled with
// optimizaion 03

__global__ void multiplyX1W1(float *X, float *W, float *Z, int m, int n){ //XXX fn_id = 2
	int global_idx = blockIdx.x * n + threadIdx.x;	
	int local_tid = threadIdx.x;
	__shared__ int adder; 
	__shared__ float instanceZxw[256];
	adder = n - blockDim.x;
	instanceZxw[local_tid] =	(X[local_tid] * W[global_idx]) + 
								(X[local_tid + 256] * W[global_idx + 256]) + 
								(X[local_tid + 512] * W[global_idx + 512]);
	__syncthreads();	
	if(local_tid + adder >= (256 * 3)){
		instanceZxw[local_tid] += (X[local_tid + adder] * W[global_idx + adder]);
	}
	__syncthreads();
	if(local_tid < 128){
		instanceZxw[local_tid] += instanceZxw[local_tid + 128];
	}
	__syncthreads();
	if(local_tid < 64){
		instanceZxw[local_tid] += instanceZxw[local_tid + 64];
	}
	__syncthreads();

	if(local_tid < 32){
		instanceZxw[local_tid] += instanceZxw[local_tid + 32];
	}
	__syncthreads();
	if(local_tid < 16){
		instanceZxw[local_tid] += instanceZxw[local_tid + 16];
	}
	__syncthreads();
	if(local_tid < 8){
		instanceZxw[local_tid] += instanceZxw[local_tid + 8];
	}
	__syncthreads();
	if(local_tid < 4){
		instanceZxw[local_tid] += instanceZxw[local_tid + 4];
	}
	__syncthreads();
	if(local_tid < 2){
		instanceZxw[local_tid] += instanceZxw[local_tid + 2];
	}
	__syncthreads();
	if(local_tid < 1){
		instanceZxw[local_tid] += instanceZxw[local_tid + 1];
	}
	__syncthreads();
	if(local_tid == 0){
		Z[blockIdx.x] = instanceZxw[0];
	}
}


// relu used in training
// implements a neuron dropout of probability 0.4
// calculates derivative wrt input z1i and saves it for backprop
// the result vector x2 is of length 1025 with each x2i shifted one
// to the right, to accomodate the bias dummy node at idx 0
// called with <<<1, 1024>>>
__global__ void reLu_train(float *z1, float *x2, float *relud, uint64_t seed){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float z1i;
	float tmp;	
	float deriv;
	float rand_num;
	curandState randinfo;
	#ifdef NRAND
	curand_init(1 + tid, 0, 0, &randinfo); // constant seed for debugging	
	#else
	curand_init(seed + tid, 0, 0, &randinfo);
	#endif
	rand_num = curand_uniform(&randinfo);
	if(rand_num <= 0.4){
		tmp = 0;
		deriv = 0;
	}
	else{
		z1i = z1[tid];		
		if(z1i < 0){
			tmp = 0;
			deriv = 0;
		}
		else{
			tmp = z1i;
			deriv = 1;
		}
	}
	__syncthreads();
	x2[tid+1] = tmp;
	relud[tid] = deriv;
	if(tid == 0) {
		x2[0] = 1; // right shift for bias node
	}
}

// used in the test run
// is the same as relu_train, but there is no chance of neuron dropout
// derivative is not needed but is not too cumbersome so it stays
// called with <<<1, 10>>>
__global__ void reLu(float *z1, float *x2, float *relud){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float z1i = z1[tid];	
	float tmp;	
	float deriv;
	if(z1i < 0){
		tmp = 0;
		deriv = 0;
	}
	else{
		tmp = z1i;
		deriv = 1;
	}
	__syncthreads();
	x2[tid+1] = tmp;
	relud[tid] = deriv;
	if(tid == 0) x2[0] = 1; // right shift for bias node
}


// see description of multiplyX1W1 for general idea because it is similar
// here X is of length 1025. idx 1-1024 are the outputs of relu 
// and idx 0 is the bias dummy node
// W is the weight vector associated with transition:
//		hidden_layer -> output_layer
// thus W is of length 10 * 1025
// the multiplication, and partial reduction is the same as multiplyX1W2
// but with the necessary changes to accomodate the different dimensions
// Z is the output array of length 10
// kernel is called <<<10, 512>>> and each block is associated with 
// one of the 10 output nodes
// called with <<<10, 512>>>
__global__ void multiplyX2W2(float *X, float *W, float *Z, int m, int n){ //XXX fn_id = 5
	int global_idx = (blockIdx.x * n) + threadIdx.x;	
	int local_tid = threadIdx.x;;
	__shared__ float sZxw[512];
	sZxw[local_tid] = 	(X[local_tid] * W[global_idx]) +  
						(X[local_tid + 512] * W[global_idx + 512]);

	__syncthreads();
	if((local_tid +	512 + 1) == 1024){
		sZxw[local_tid] += (X[local_tid + 512 + 1] * W[global_idx + 512 + 1]);
	}	
	__syncthreads();

	if(local_tid < 256){
		sZxw[local_tid] += sZxw[local_tid + 256];
	}
	__syncthreads();	
	
	if(local_tid < 128){
		sZxw[local_tid] += sZxw[local_tid + 128];
	}
	__syncthreads();
	if(local_tid < 64){
		sZxw[local_tid] += sZxw[local_tid + 64];
	}
	__syncthreads();

	if(local_tid < 32){
		sZxw[local_tid] += sZxw[local_tid + 32];
	}
	__syncthreads();
	if(local_tid < 16){
		sZxw[local_tid] += sZxw[local_tid + 16];
	}
	__syncthreads();
	if(local_tid < 8){
		sZxw[local_tid] += sZxw[local_tid + 8];
	}
	__syncthreads();
	if(local_tid < 4){
		sZxw[local_tid] += sZxw[local_tid + 4];
	}
	__syncthreads();
	if(local_tid < 2){
		sZxw[local_tid] += sZxw[local_tid + 2];
	}
	__syncthreads();
	if(local_tid < 1){
		sZxw[local_tid] += sZxw[local_tid + 1];
	}
	__syncthreads();
	if(local_tid == 0){
		Z[blockIdx.x] = sZxw[0];
	}

}

// brings in model predicted result coming from the hidden layer z2, 
// computes softmax and holds result in y_soft_predict
// one thread computes softmax, and protects against over/underflow
// by finding the max of z2 and subtracts that from each of the 10
// prediction values, and proceeds with the apropriate calculations 
// involving e, and using the sum as the denominator
// called with <<<1, 10>>>
__global__ void softmax(float* z2, float *y_soft_predict){
	int tid = threadIdx.x;
	__shared__ float z2_local[NUM_LABELS];
	__shared__ float sum;	
	float tmp;	
	z2_local[tid] = z2[tid];
	__syncthreads();
	if(tid == 0){
		float max = -10000.0;
		for(int i =0; i < NUM_LABELS; i++){
			if(max < z2_local[i]) max = z2_local[i];
		}	
		sum = 0.0;		
		for(int i = 0; i < NUM_LABELS; i++){
			tmp = exp(z2_local[i] - max);			
			z2_local[i] = tmp;
			sum += z2_local[i];
		}
	}
	__syncthreads();
	y_soft_predict[tid] = z2_local[tid] / sum;
}


// calculates cross entropy
// calculates the derivative of cross entropy wrt y_soft_predict
// calculates the derivative of cross entropy wrt z2 and held in z2d
// prints batch entropy after each batch if compiled with -DPBATCH
// prints epoch entropy after each epoch 
// explanation of macro definitions are in the makefile
// epoch entropy is slightly misleading because it is actually the average 
// loss of each instance trained, thus what is printed is more representative
// of a configuration where the batch size is the size of the training set/
// But this is only relevant to what is printed to the terminal.
// Everything that takes place within the function is accurate to the model design.
// Batch entropy printed to the terminal is more representative of what is 
// taking place within the model
// aside from the calculation of z2d, most of the work is done by
// one thread. Although this may be inefficient, the idea is to save 
// comput on memory operations transfering from device to host
// z2d is calculated by subtracting the one hot y_true label vector
// from the model prediction vector Y_soft_predict
// called with <<<1, 10>>>
__global__ void z2d_and_ce(float* y_soft_predict, unsigned char* y_true, float* z2d,
												float* epoch_ce, float* batch_ce, 
												int instance_idx, int batch_size, 
												int batch_idx, int num_batches){
	int tid = threadIdx.x;
	float ce;
	__shared__ float both_local[NUM_LABELS * 2];
	both_local[tid] = y_soft_predict[tid];
	both_local[tid + NUM_LABELS] = (float)(y_true[tid]);
	__syncthreads();	
	z2d[tid] = both_local[tid] - both_local[tid + NUM_LABELS];
	if(both_local[tid + NUM_LABELS]){
		ce = -(log(both_local[tid]));	
		*batch_ce += ce;
		*epoch_ce += ce;
	}
	__syncthreads();	
	if(tid == 0){	
		if((batch_idx+1 == num_batches) && (instance_idx+1 == batch_size)){
			#ifdef PBATCH			
			printf(".................................. Batch Loss: %f\n\n", (*batch_ce) / batch_size);			
			#endif
			printf(".................................. Epoch Loss: %f\n\n", (*epoch_ce) / NUM_TRAIN_INSTANCES);
			*batch_ce = 0.0;
			*epoch_ce = 0.0;
		}	
		else if(instance_idx+1 == batch_size){ 
			#ifdef PBATCH
			printf(".................................. Batch Loss: %f\n\n", (*batch_ce) / batch_size); 
			#endif
			*batch_ce = 0.0;
		}
	}
}



// calculates the derivative of cross entropy wrt w2 weight matrix
// associated with transition hidden_neurons -> output_layer
// w2d = x2-transposed * z2d
// x2 is only transposed logically via implementation 
// x2 is of lenth 1025
// z2d is of length 10
// Thus the resulting matrix is the same size as w2 .. 10 * 1025.
// Each thread is in charge of calculating one column of w2d.
// Thread 0 - 9 calculate the remaining 1025th column.
// Calculations are done via loop unroll.
// w2d is zeroed out at the beginning of a batch.
// And as the batch proceeds w2d accumulates the values from each instance
// processed, and is used to calculate the average gradient upon
// batch completion
// called with <<<1, 1024>>>
__global__ void calcW2d(float *X2, float *z2d, float *w2d, int m_w2d, int n_w2d){
	int local_tid = threadIdx.x;
	float tmp_w2id;	
	int w2d_idx;
	__shared__ float sz2d[NUM_LABELS];
	__shared__ float sx2[HIDDEN_NEURONS + 1];
	sx2[local_tid] = X2[local_tid];	
	if(local_tid < NUM_LABELS){
		sz2d[local_tid] = z2d[local_tid];
	}
	if(local_tid == 0){
		sx2[HIDDEN_NEURONS] = X2[HIDDEN_NEURONS];
	}
	__syncthreads();
	 
	tmp_w2id = sx2[local_tid] * sz2d[0];
	__syncthreads();	
	w2d[local_tid] += tmp_w2id;

	w2d_idx = (n_w2d * 1) + local_tid;
	tmp_w2id = sx2[local_tid] * sz2d[1];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;
	
	w2d_idx = (n_w2d * 2) + local_tid;	
	tmp_w2id = sx2[local_tid] * sz2d[2];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 3) + local_tid;	
	tmp_w2id = sx2[local_tid] * sz2d[3];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 4) + local_tid;	
	tmp_w2id = sx2[local_tid] * sz2d[4];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 5) + local_tid;	
	tmp_w2id = sx2[local_tid] * sz2d[5];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 6) + local_tid;
	tmp_w2id = sx2[local_tid] * sz2d[6];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 7) + local_tid;
	tmp_w2id = sx2[local_tid] * sz2d[7];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 8) + local_tid;
	tmp_w2id = sx2[local_tid] * sz2d[8];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	w2d_idx = (n_w2d * 9) + local_tid;
	tmp_w2id = sx2[local_tid] * sz2d[9];
	__syncthreads();	
	w2d[w2d_idx] += tmp_w2id;

	if(local_tid < NUM_LABELS){
		w2d_idx = (n_w2d * local_tid) + HIDDEN_NEURONS;
		tmp_w2id = sx2[HIDDEN_NEURONS] * sz2d[local_tid];
		__syncthreads();
		w2d[w2d_idx] += tmp_w2id;
	}
}


// Calculates the derivative of cross entropy wrt x2 in the process
// of calculating the derivative of cross entropy wrt z1
// x2d is z2d * weight matrix w2 transposed
// This is calculated in the running sum during the loop unroll
// x2d is of length 1025 but we do not care about the bias dummy at idx 0
// this is why 1 is added to the global idx in each unrolled loop iteration
// z1d is the tensor product of x2d(not including dummy bias) and
// the derivative of relu wrt z1.
// the derivative of relu is calculated during the forward pass
// and is used here in the last line to compute z1d
// zid is of length 1024 and thus each thread calculates one idx
// Called with <<<1, 1024>>> 
__global__ void calcz1d(float* z2d, float* W2, float* z1d, float* relud, int m_w2, int n_w2){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int local_tid = threadIdx.x;
	float sum = 0;
	__shared__ float sz2d[NUM_LABELS];

	if(local_tid < NUM_LABELS){
		sz2d[local_tid] = z2d[local_tid];
	}
	__syncthreads();

	sum += (sz2d[0] * W2[global_tid+1]);
	__syncthreads();
	sum += (sz2d[1] * W2[(n_w2 * 1) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[2] * W2[(n_w2 * 2) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[3] * W2[(n_w2 * 3) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[4] * W2[(n_w2 * 4) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[5] * W2[(n_w2 * 5) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[6] * W2[(n_w2 * 6) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[7] * W2[(n_w2 * 7) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[8] * W2[(n_w2 * 8) + global_tid+1]);
	__syncthreads();
	sum += (sz2d[9] * W2[(n_w2 * 9) + global_tid+1]);
	__syncthreads();
	z1d[global_tid] = relud[global_tid] * sum; // now this vector is shifted left, with len 1024
}



// Calculates the derivative of cross entropy wrt weight matrix w1
// associated with the transition input_layer -> hidden layer
// w1d = input instance x1 transposed * z1d
// Called with <<<32, 1024>>> which was determined after a few performance test
// runs with different kernel block parameters
// Operands are held in block shared arrays.
// Calculations are done in a for loop using a strided increment to use 
// efficient coalesced memory writes.
// Like w2d, the values accumulate over a batch to compute average gradient
__global__ void calcW1d(float *X1, float *z1d, float *w1d, int m_w1d, int n_w1d){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int local_tid = threadIdx.x;
	__shared__ float sX[NUM_FEATURES + 1];
	__shared__ float sZ1d[HIDDEN_NEURONS];
	sZ1d[local_tid] = z1d[local_tid];	
	if(local_tid < (NUM_FEATURES + 1)){
		sX[local_tid] = X1[local_tid];
	}
	__syncthreads();
	int stride = blockDim.x * gridDim.x;
	int num_weights = m_w1d * n_w1d;
	int xi;
	int z1di;
	for(int idx = global_tid; idx < num_weights; idx += stride){
		xi = idx % (NUM_FEATURES + 1);
		z1di = idx % HIDDEN_NEURONS;
		w1d[idx] += sX[xi] * sZ1d[z1di]; 
		__syncthreads();
	}
}


// updates weight matrix w1 upon batch completion.
// Each value is the accumulated sum of w1di over a batch.
// This sum is divided by batch size to compute the average, and this value
// is subtracted from the associated weight with a magnitude determined by multiplying
// with the learning rate.
// Each value of w2d is zeroed out for the next batch
// w1 and w1d are of size 785 * 1024
// called with <<<32, 1024>>>
__global__ void updateW1(float* w1, float* w1d, long batch_size, float learning_rate){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	float w_change;
	int stride = blockDim.x * gridDim.x;
	int num_weights = (NUM_FEATURES + 1) * HIDDEN_NEURONS;
	for(int idx = global_tid; idx < num_weights; idx += stride){
		//printf("idx %d\n", idx);		
		w_change = (w1d[idx] / (float)batch_size) * learning_rate;		
		__syncthreads();
		w1d[idx] = 0.0;
		w1[idx] -= w_change;

	}
	__syncthreads();
} 

// Updates weight matrix w2 upon batch completion
// Similar to updateW1 but updates in unrolled loop.
// w2 and w2d are of size 10 * 1025
// called with <<<1, 1024>>>
__global__ void updateW2(float* w2, float* w2d, long batch_size, float learning_rate){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;		
	float w_change;												
	
	w_change = (w2d[global_tid]  / (float)batch_size) * learning_rate;
	__syncthreads();
	w2d[global_tid] = 0.0;
	w2[global_tid] -= w_change;
	
	w_change = (w2d[((HIDDEN_NEURONS + 1) * 1) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 1) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 1) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 2) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 2) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 2) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 3) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 3) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 3) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 4) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 4) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 4) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 5) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 5) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 5) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 6) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 6) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 6) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 7) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 7) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 7) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 8) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 8) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 8) + global_tid] -= w_change;

	w_change = (w2d[((HIDDEN_NEURONS + 1) * 9) + global_tid] / (float)batch_size) * learning_rate;
	__syncthreads();	
	w2d[((HIDDEN_NEURONS + 1) * 9) + global_tid] = 0.0;
	w2[((HIDDEN_NEURONS + 1) * 9) + global_tid] -= w_change;
	
	if(global_tid < NUM_LABELS){
		w_change = (w2d[((HIDDEN_NEURONS + 1) * global_tid) + HIDDEN_NEURONS] / (float)batch_size) * learning_rate;
		__syncthreads();
		w2d[((HIDDEN_NEURONS + 1) * global_tid) + HIDDEN_NEURONS] = 0.0;
		w2[((HIDDEN_NEURONS + 1) * global_tid) + HIDDEN_NEURONS] -= w_change;
	}
}


// tallies number of correct predictions in the test run
__global__ void test_run(float* y_soft_predict, unsigned char * y_true, int * num_correct){
	int tid = threadIdx.x;
	__shared__ float sy_predict[NUM_LABELS];
	int idx = -1;	
	sy_predict[tid] = y_soft_predict[tid];
	__syncthreads();	
	if(tid == 0){
		float max = -10000.0;
		for(int i =0; i < NUM_LABELS; i++){
			if(max < sy_predict[i]) {
				max = sy_predict[i];
				idx = i;			
			}
		}		
		if(y_true[idx]){
			(*num_correct)++;
		}
	}
}


// kernels that are no longer in use -------------------------------------------
// -----------------------------------------------------------------------------


#if 0
__global__ void multiplyX1W1(float *X, float *W, float *Zxw, int m, int n){ //XXX fn_id = 1
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int local_tid = threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int num_weights = m * n;
	int xi;	
	__shared__ float sX[NUM_FEATURES + 1];

	if(local_tid < (NUM_FEATURES + 1)){
		sX[local_tid] = X[local_tid];
	}
	__syncthreads();
	for(int idx = global_tid; idx < num_weights; idx += stride){
		xi = idx % (NUM_FEATURES + 1);
		Zxw[idx] = sX[xi] * W[idx]; 
	}
}
#endif


#if 0
__global__ void reduceZ1(float *Zxw, float *Z, int m_Zxw, int n_Zxw){ //XXX fn_id = 1
	__shared__ int adder;
	__shared__ float instanceZxw[256];		
	int global_idx = blockIdx.x * n_Zxw + threadIdx.x;
	int local_tid = threadIdx.x;	     
	adder = n_Zxw - blockDim.x;	
	instanceZxw[local_tid] = Zxw[global_idx] + Zxw[global_idx + 256] + Zxw[global_idx + 512];
	__syncthreads();
	if((local_tid + adder) >= (256 * 3)){  //&& ((local_tid + adder) < n_Zxw)){
		instanceZxw[local_tid] += Zxw[global_idx + adder];
	}
	__syncthreads();
	if(local_tid < 128){
		instanceZxw[local_tid] += instanceZxw[local_tid + 128];
	}
	__syncthreads();
	if(local_tid < 64){
		instanceZxw[local_tid] += instanceZxw[local_tid + 64];
	}
	__syncthreads();

	if(local_tid < 32){
		instanceZxw[local_tid] += instanceZxw[local_tid + 32];
	}
	__syncthreads();
	if(local_tid < 16){
		instanceZxw[local_tid] += instanceZxw[local_tid + 16];
	}
	__syncthreads();
	if(local_tid < 8){
		instanceZxw[local_tid] += instanceZxw[local_tid + 8];
	}
	__syncthreads();
	if(local_tid < 4){
		instanceZxw[local_tid] += instanceZxw[local_tid + 4];
	}
	__syncthreads();
	if(local_tid < 2){
		instanceZxw[local_tid] += instanceZxw[local_tid + 2];
	}
	__syncthreads();
	if(local_tid < 1){
		instanceZxw[local_tid] += instanceZxw[local_tid + 1];
	}
	__syncthreads();

	if(local_tid == 0){
		Z[blockIdx.x] = instanceZxw[0];
	}
}
#endif

#if 0
__global__ void reduce(float *Zxw, float *Z, int m_Zxw, int n_Zxw){
	__shared__ int adder; // XXX i dont think this needs to be shared anymore
	extern __shared__ float instanceZxw[];		
	int global_idx = blockIdx.x * n_Zxw + threadIdx.x;
	int local_tid = threadIdx.x;	     
	adder = n_Zxw - blockDim.x;	
	instanceZxw[local_tid] = Zxw[global_idx];
	
	if(((local_tid + adder) >= blockDim.x) && ((local_tid + adder) < n_Zxw)){
		instanceZxw[local_tid + adder] = Zxw[global_idx + adder];
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(local_tid < stride){
			instanceZxw[adder + local_tid] += instanceZxw[adder + local_tid + stride];
		}
		__syncthreads();
	}
	__syncthreads();
	if(adder){  // HARD CODED CASES because I couldnt figure out a generalization for now
		if(local_tid == 0){ // there are only a few specific cases so no big deal
			instanceZxw[adder-1] += instanceZxw[adder];
		}
		__syncthreads();
		if(adder == 273){			
			int new_adder = adder - 256;
			for(unsigned int stride = 256/2; stride > 0; stride >>= 1){
				if(local_tid < stride){
					instanceZxw[new_adder + local_tid] += instanceZxw[new_adder + local_tid + stride];
				}	
				__syncthreads();
			}
			__syncthreads();
			if(local_tid == 0){
				instanceZxw[new_adder-1] += instanceZxw[new_adder];
				instanceZxw[new_adder-2] += instanceZxw[new_adder - 1];
			}
			if(local_tid < 8){ // warp reduce
				instanceZxw[local_tid] += instanceZxw[local_tid + 8];
				instanceZxw[local_tid] += instanceZxw[local_tid + 4];
				instanceZxw[local_tid] += instanceZxw[local_tid + 2];
				instanceZxw[local_tid] += instanceZxw[local_tid + 1];
			}
		} 
		else if(adder == 2){
			if(local_tid == 0) {
				instanceZxw[0] += instanceZxw[1];
			}
		} 
	}
	__syncthreads();
	if(local_tid == 0){
		Z[blockIdx.x] = instanceZxw[0];
	}
}
#endif

#if 0
__global__ void reduceZ1(float *Zxw, float *Z, int m_Zxw, int n_Zxw){
	__shared__ int adder;
	extern __shared__ float instanceZxw[];		
	int global_idx = blockIdx.x * n_Zxw + threadIdx.x;
	int local_tid = threadIdx.x;	     
	adder = n_Zxw - blockDim.x;	
	instanceZxw[local_tid] = Zxw[global_idx];
	
	if(((local_tid + adder) >= blockDim.x) && ((local_tid + adder) < n_Zxw)){
		instanceZxw[local_tid] += Zxw[global_idx + adder];
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(local_tid < stride){
			instanceZxw[local_tid] += instanceZxw[local_tid + stride];
		}
		__syncthreads();
	}
	__syncthreads();

	if(local_tid == 0){
		Z[blockIdx.x] = instanceZxw[0];
	}
}
#endif

#if 0
__global__ void multiplyX2W2(float *X, float *W, float *Z, int m, int n){ // XXX fn_id = 3
	int global_idx = (blockIdx.x * n) + threadIdx.x;	
	int local_tid = threadIdx.x;;
	__shared__ float sX[HIDDEN_NEURONS + 1];
	__shared__ float sZxw[HIDDEN_NEURONS + 1];
	sX[local_tid] = X[local_tid];
	if(local_tid == 0){
		sX[HIDDEN_NEURONS] = X[HIDDEN_NEURONS];
	}
	__syncthreads();
	
	sZxw[local_tid] = sX[local_tid] * W[global_idx];
	__syncthreads();
	if(local_tid == 0) {
		sZxw[local_tid] += X[HIDDEN_NEURONS] * W[global_idx + HIDDEN_NEURONS];
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(local_tid < stride){
			sZxw[local_tid] += sZxw[local_tid + stride];
		}
		__syncthreads();
	}
	__syncthreads();
	if(local_tid == 0){
		Z[blockIdx.x] = sZxw[0]; // this is actually Z not Zxw
	}

}
#endif

#if 0
__global__ void multiplyX2W2(float *X, float *W, float *Z, int m, int n){ //XXX fn_id = 4
	int global_idx = (blockIdx.x * n) + threadIdx.x;	
	int local_tid = threadIdx.x;;
	//__shared__ float sX[HIDDEN_NEURONS + 1];
	__shared__ float sZxw[256];
	sZxw[local_tid] = 	(X[local_tid] * W[global_idx]) + 
						(X[local_tid + 256] * W[global_idx + 256]) + 
						(X[local_tid + 512] * W[global_idx + 512]) +
						(X[local_tid + 768] * W[global_idx + 768]);

	__syncthreads();
	if((local_tid + 768 + 1) == 1024){
		sZxw[local_tid] += (X[local_tid + 768 + 1] * W[global_idx + 768 + 1]);
	}	
	__syncthreads();

	
	if(local_tid < 128){
		sZxw[local_tid] += sZxw[local_tid + 128];
	}
	__syncthreads();
	if(local_tid < 64){
		sZxw[local_tid] += sZxw[local_tid + 64];
	}
	__syncthreads();

	if(local_tid < 32){
		sZxw[local_tid] += sZxw[local_tid + 32];
	}
	__syncthreads();
	if(local_tid < 16){
		sZxw[local_tid] += sZxw[local_tid + 16];
	}
	__syncthreads();
	if(local_tid < 8){
		sZxw[local_tid] += sZxw[local_tid + 8];
	}
	__syncthreads();
	if(local_tid < 4){
		sZxw[local_tid] += sZxw[local_tid + 4];
	}
	__syncthreads();
	if(local_tid < 2){
		sZxw[local_tid] += sZxw[local_tid + 2];
	}
	__syncthreads();
	if(local_tid < 1){
		sZxw[local_tid] += sZxw[local_tid + 1];
	}
	__syncthreads();
	if(local_tid == 0){
		Z[blockIdx.x] = sZxw[0];
	}

}
#endif

#if 0
__global__ void reduceZ2(float *Zxw, float *Z, int m_Zxw, int n_Zxw){
	__shared__ int adder; // XXX i dont think this needs to be shared anymore
	extern __shared__ float instanceZxw[];		
	int global_idx = blockIdx.x * n_Zxw + threadIdx.x;
	int local_tid = threadIdx.x;	     
	adder = n_Zxw - blockDim.x;	
	instanceZxw[local_tid] = Zxw[global_idx];
	
	if(((local_tid + adder) >= blockDim.x) && ((local_tid + adder) < n_Zxw)){
		instanceZxw[local_tid] += Zxw[global_idx + adder];
		//printf("blocdidx %d   local_tid %d  adder %d\n", blockIdx.x, local_tid, adder);
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(local_tid < stride){
			instanceZxw[local_tid] += instanceZxw[local_tid + stride];
		}
		__syncthreads();
	}
	__syncthreads();

	if(local_tid == 0){
		Z[blockIdx.x] = instanceZxw[0];
	}
}
#endif

#if 0
__global__ void multiplyX2W2(float *X, float *W, float *Zxw, int m, int n){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int local_tid = threadIdx.x;
	__shared__ float sX[HIDDEN_NEURONS + 1];
	sX[local_tid] = X[local_tid];	
	if(local_tid == 0){
		sX[HIDDEN_NEURONS] = X[HIDDEN_NEURONS];
	}
	__syncthreads();
	int stride = blockDim.x * gridDim.x;
	int num_weights = m * n;
	int xi;
	for(int idx = global_tid; idx < num_weights; idx += stride){
		xi = idx % (HIDDEN_NEURONS + 1);
		Zxw[idx] = sX[xi] * W[idx]; 
	}
	if(global_tid < m){
		Zxw[(global_tid * n) + HIDDEN_NEURONS] = sX[HIDDEN_NEURONS] * W[(global_tid * n) + HIDDEN_NEURONS];
	}
}
#endif

#if 0
__global__ void multiplyX2W2(float *X, float *W, float *Zxw, int m, int n){
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;	
	//int local_tid = threadIdx.x;;
	float xi = X[global_tid];
	Zxw[global_tid] = xi * W[global_tid];
	Zxw[(n * 1) + global_tid] = xi * W[(n * 1) + global_tid];
	Zxw[(n * 2) + global_tid] = xi * W[(n * 2) + global_tid];
	Zxw[(n * 3) + global_tid] = xi * W[(n * 3) + global_tid];
	Zxw[(n * 4) + global_tid] = xi * W[(n * 4) + global_tid];
	Zxw[(n * 5) + global_tid] = xi * W[(n * 5) + global_tid];
	Zxw[(n * 6) + global_tid] = xi * W[(n * 6) + global_tid];
	Zxw[(n * 7) + global_tid] = xi * W[(n * 7) + global_tid];
	Zxw[(n * 8) + global_tid] = xi * W[(n * 8) + global_tid];
	Zxw[(n * 9) + global_tid] = xi * W[(n * 9) + global_tid];
	if(global_tid < NUM_LABELS) {
		Zxw[(n * global_tid) + HIDDEN_NEURONS] = X[HIDDEN_NEURONS] * W[(n * global_tid) + HIDDEN_NEURONS];
	}
}
#endif



// model verification functions ------------------------------------------------

__global__ void mulcheck(float *X, float *W, float *Zxw, int m, int n){
	int num_weights = m * n;
	float zxiwii;	
	float xi;
	float wii;	
	float prod;
	for(int idx = 0; idx < num_weights; idx++){
							//for(int idx = 0; idx < (NUM_FEATURES + 1); idx++){ // use to check just one hidden neuron
		zxiwii = Zxw[idx];
							//xi = X[idx %(NUM_FEATURES+1)];
		xi = X[idx % n]; 	// changed (num_featurees + 1) to n
		wii = W[idx];
		prod = xi * wii;
							//printf("MULcheck Zxw[%d](%f)......check against = %f\n", idx, Zxw[idx], X[idx %(NUM_FEATURES+1)] * W[idx]);
		printf("MULcheck Zxw[%d](%f)......check against = %f........xi[%d] %f  wii[%d] %f\n", idx, zxiwii, prod, idx, xi, idx, wii);
							//assert(abs(Zxw[idx] - (X[idx % (NUM_FEATURES+1)] * W[idx])) < (float)1E-6);
		assert(abs(Zxw[idx] - prod) < (float)1E-6);
	}
}


__global__ void xicheck(float* X, int len){
	printf("xicheck\n");	
	for(int i = 0; i < len; i++){
		printf("dX[%d] = %f\t", i, X[i]);
			if((i%10) == 0) printf("\n");
	}
}


__global__ void reducecheck(float *Zxw, float *Z, int m_Zxw, int n_Zxw){
	float sum;
	float zi;
	for(int i = 0; i < m_Zxw; i++){
		sum = 0;		
		for(int j = 0; j < n_Zxw; j++){
			sum += Zxw[(n_Zxw * i) + j];
		}
		zi = Z[i];		
		printf("REDcheck Z[%d](%f)......check against = %f\n", i, zi, sum);
		assert(abs(zi - sum) < (float)1E-4);
	}
}

__global__ void reducemulcheck(float *X, float *W, float *Z, int m, int n){
	float sum;
	float zi;
	for(int i = 0; i < m; i++){
		sum = 0;		
		for(int j = 0; j < n; j++){
			sum += (X[j] * W[(n * i) + j]);
		}
		zi = Z[i];		
		printf("REDMULcheck Z[%d](%f)......check against = %f\n", i, zi, sum);
		assert(abs(zi - sum) < (float)1E-4);
	}
}


__global__ void transposemulcheck(float *X2, float *z2d, float *w2d, float *w2d_accum, int m_w2d, int n_w2d){
	float x2iz2di;	
	float xi;
	float z2di;	
	float prod;
	
	for(int i = 0; i < m_w2d; i++){
		z2di = z2d[i];
		for(int j = 0; j < n_w2d; j++){
			x2iz2di = w2d[(n_w2d * i) + j];
			xi = X2[j];
			prod = xi * z2di;
			printf("transMulcheck w2d[%d](%f)......check against = %f........xi[%d] %f  z2di[%d] %f\n", (n_w2d * i) + j, x2iz2di, prod, j, xi, i, z2di);
			assert(abs(x2iz2di - prod) < (float)1E-6);
		}
	}
}


__global__ void x2dreducecheck(float* z2d, float* W2, float* x2dz2dw2, float* x2d, int m_x2dz2dw2, int n_x2dz2dw2){
	float sum;
	float x2di;
	for(int i = 1; i < n_x2dz2dw2; i++){
		sum = 0;		
		for(int j = 0; j < m_x2dz2dw2; j++){
			sum += z2d[j] * W2[(n_x2dz2dw2 * j) + i];
		}
		x2di = x2d[i-1];		
		printf("x2dREDcheck x2d[%d](%f)......check against = %f\n", i-1, x2d[i-1], sum);
		assert(abs(x2di - sum) < (float)1E-4);
	}
}



// END OF NEURAL NET CODE
// these were some experiments to get used to cuda thread orgainzation
// -----------------------------------------------------------------------------

__global__ void test(){
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	printf("TID: %d\tx: %d\ty: %d\n", tid, x, y);
}


__global__ void indexing_test(){ //XXX not working how I want, but I will move on

	int tid_1D_1D = blockIdx.x * blockDim.x + threadIdx.x;

	int tid_1D_2D = blockIdx.x * blockDim.x * blockDim.y
			+ threadIdx.y * blockDim.x + threadIdx.x;

	int tid_1D_3D = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x
			+ threadIdx.y * blockDim.x + threadIdx.x;


	int bid_2D_1D = blockIdx.y * gridDim.x + blockIdx.x;
	int tid_2D_1D = bid_2D_1D * blockDim.x + threadIdx.x;


	int bid_2D_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int tid_2D_2D = bid_2D_2D * (blockDim.x * blockDim.y) 
			+ (threadIdx.y * blockDim.x) + threadIdx.x;


	int bid_2D_3D = blockIdx.x + blockIdx.y * gridDim.x;
	int tid_2D_3D = bid_2D_3D * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;


	int bid_3D_1D = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int tid_3D_1D = bid_3D_1D * blockDim.x + threadIdx.x;


	int bid_3D_2D = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int tid_3D_2D = bid_3D_2D * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;


	int bid_3D_3D = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int tid_3D_3D = bid_3D_3D * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	printf("---------------------------------------------------------------\n\n");
	printf("gridDim.z\t%d\tblockDim.z\t%d\n", gridDim.z, blockDim.z);
	printf("gridDim.y\t%d\tblockDim.y\t%d\n", gridDim.y, blockDim.y);
	printf("gridDim.x\t%d\tblockDim.x\t%d\n", gridDim.x, blockDim.x);
	printf("1D_1D_tid: %d\n", tid_1D_1D);
	printf("1D_2D_tid: %d\n", tid_1D_2D);
	printf("1D_3D_tid: %d\n", tid_1D_3D);
	printf("2D_1D_tid: %d\n", tid_2D_1D);
	printf("2D_2D_tid: %d\n", tid_2D_2D);
	printf("2D_3D_tid: %d\n", tid_2D_3D);
	printf("3D_1D_tid: %d\n", tid_3D_1D);
	printf("3D_2D_tid: %d\n", tid_3D_2D);
	printf("3D_3D_tid: %d\n\n", tid_3D_3D);

	printf("2D_1D_bid: %d\n", bid_2D_1D);
	printf("2D_2D_bid: %d\n", bid_2D_2D);
	printf("2D_3D_bid: %d\n", bid_2D_3D);
	printf("3D_1D_bid: %d\n", bid_3D_1D);
	printf("3D_2D_bid: %d\n", bid_3D_2D);
	printf("3D_3D_bid: %d\n\n", bid_3D_3D);
	printf("---------------------------------------------------------------\n");
}


__global__ void print_w(float* W, int m, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid == 0){
		printf("print_w\n");
	}
	for(int m_idx = 0; m_idx < m; m_idx++){
		for(int n_idx = 0; n_idx < n; n_idx++){
			printf("{W[%d][%d]: %f } ", m_idx, n_idx, W[m_idx*n + n_idx]);
		}
		printf("\n");
	}
	printf("W[%d][%d]\n", m, n);
}







