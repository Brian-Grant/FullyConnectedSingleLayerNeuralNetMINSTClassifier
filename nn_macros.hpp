#ifndef NN_MACROS_HPP
#define NN_MACROS_HPP


// -----------------------------------------------------------------------------
// macros for input data file parsing-------------------------------------------
#define TRAIN_FEATURES "mnist/train-images.idx3-ubyte"
#define TRAIN_LABELS "mnist/train-labels.idx1-ubyte"
#define TEST_FEATURES "mnist/t10k-images.idx3-ubyte"
#define TEST_LABELS "mnist/t10k-labels.idx1-ubyte"

#define FEATURE_MAGIC 2051
#define LABEL_MAGIC 2049

// -----------------------------------------------------------------------------
// macros related to the model and the input data-------------------------------
#define NUM_FEATURES 28*28 // 784 
#define NUM_LABELS 10

#define NUM_TRAIN_INSTANCES 60000
#define NUM_TEST_INSTANCES 10000
#define HIDDEN_NEURONS 1024
#define IMAGE_DIM 28

#define WARP_SIZE 32

#define INSTANCE_BYTES (NUM_FEATURES + 1) * sizeof(float)

// -----------------------------------------------------------------------------
// macros for cuda error checking-----------------------------------------------
#ifdef GPUD
#define gpuAssert(ans) { errorCheck((ans), __FILE__, __LINE__); }
inline void errorCheck(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);	
	}
}

#define syncPeek() { cudaError_t cuda_rv = cudaDeviceSynchronize(); 	\
					gpuAssert( cuda_rv ); 								\
					cuda_rv = cudaPeekAtLastError();					\
					gpuAssert( cuda_rv ); }	

#else
#define gpuAssert(ans) ans = ans
#define syncPeek()
#endif

// -----------------------------------------------------------------------------
// macros to time kernel execution times----------------------------------------
#define START_TIMER(xx)	float timed##xx;							\
						cudaEvent_t start##xx, stop##xx;			\
						cudaEventCreate(&start##xx);				\
    					cudaEventCreate(&stop##xx);					\
						cudaEventRecord(start##xx, 0)			

#define STOP_TIMER(xx) 	cudaEventRecord(stop##xx, 0);							\
						cudaEventSynchronize(stop##xx);							\
						cudaEventElapsedTime(&timed##xx, start##xx, stop##xx);	\
						printf("TIMER %d:  %f s .... (%f ms) \n", 				\
								xx, timed##xx / 1000.0f, timed##xx)



#endif
