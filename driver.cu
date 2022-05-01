#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#include "nn_macros.hpp"
#include "cuda_fns.hpp"
#include "mnist_parser.hpp"

void test_wrapper();
void index_test_wrapper();


void get_hyper_parameters(float& LEARNING_RATE, long& EPOCHS, long& BATCH_SIZE, long& NUM_BATCHES, char** argv);

int main(int argc, char** argv){
	std::cout <<  std::endl;
	

	if(argc != 4){
		std::cout << "USAGE: ./mnist-nn <learning_rate> <epochs> <batch_size>\n" << std::endl;
		exit(0);
	}	
	
	std::cout << "FULLY CONNECTED NEURAL NETWORK MNIST CLASSIFIER" << std::endl;	
	
	static float 			train_feature_set[NUM_TRAIN_INSTANCES][NUM_FEATURES+1];
	static unsigned char 	train_label_set[NUM_TRAIN_INSTANCES][NUM_LABELS] = {0};
	
	static float 			test_feature_set[NUM_TEST_INSTANCES][NUM_FEATURES+1];
	static unsigned char 	test_label_set[NUM_TEST_INSTANCES][NUM_LABELS] = {0};
	
	parse_features	(TRAIN_FEATURES, train_feature_set);
	parse_labels	(TRAIN_LABELS, train_label_set);

	parse_features	(TEST_FEATURES, test_feature_set);
	parse_labels	(TEST_LABELS, test_label_set);
							/*
								// prints out digit with ascii representing the pixels 
								print_set(train_feature_set, train_label_set);
								print_set(test_feature_set, test_label_set);
								exit(0);
							*/ 
	std::cout << "\nBuilding model..........\n" << std::endl;
	
	float LEARNING_RATE;
	long EPOCHS;
	long BATCH_SIZE; 
	long NUM_BATCHES;
	
	get_hyper_parameters(LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_BATCHES, argv);
	
	cudaError_t cuda_rv;
	
	float* x1_batch; // transfer a batch of (instances + bias) at a time
	size_t gpu_malloc_size = BATCH_SIZE * (NUM_FEATURES + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&x1_batch, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float* w1; // weights. input_layer -> hidden_layer
	gpu_malloc_size = HIDDEN_NEURONS * (NUM_FEATURES + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&w1, gpu_malloc_size);
	gpuAssert(cuda_rv);

	float* z1; // the product of x1i * w1 after the reduction
	gpu_malloc_size =  1 * HIDDEN_NEURONS * sizeof(float);
	cuda_rv = cudaMalloc(&z1, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float* relud; // holds the derivative of relu( z1i )
	gpu_malloc_size =  1 * HIDDEN_NEURONS * sizeof(float);
	cuda_rv = cudaMalloc(&relud, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float* x2; // the result of relu( z1i ) + bias 
	gpu_malloc_size = 1 * (HIDDEN_NEURONS + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&x2, gpu_malloc_size);
	gpuAssert(cuda_rv);

	float* w2; // weights.. hidden_layer -> output_layer
	gpu_malloc_size = NUM_LABELS * (HIDDEN_NEURONS + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&w2, gpu_malloc_size);
	gpuAssert(cuda_rv);

	float* z2; // the product of x2i * w2 after reduction
	gpu_malloc_size = 1 * NUM_LABELS * sizeof(float);
	cuda_rv = cudaMalloc(&z2, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float* y_soft_predict; // softmax( z2 )
	gpu_malloc_size = 1 * NUM_LABELS * sizeof(float);
	cuda_rv = cudaMalloc(&y_soft_predict, gpu_malloc_size);
	gpuAssert(cuda_rv);	
	
	unsigned char* y_true_batch; // batch of labels associated with instances
	gpu_malloc_size = BATCH_SIZE * NUM_LABELS * sizeof(unsigned char);
	cuda_rv = cudaMalloc(&y_true_batch, gpu_malloc_size);
	gpuAssert(cuda_rv);	

	float *epoch_ce; // cumulative loss per epoch
	gpu_malloc_size = sizeof(float);
	cuda_rv = cudaMalloc(&epoch_ce, gpu_malloc_size);
	gpuAssert(cuda_rv);

	float *batch_ce; // cumulative loss per batch
	gpu_malloc_size = sizeof(float);
	cuda_rv = cudaMalloc(&batch_ce, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float init_ce = 0.0; // initialize both ce variables
	cuda_rv = cudaMemcpy(epoch_ce, &init_ce, sizeof(float), cudaMemcpyHostToDevice);  
	gpuAssert(cuda_rv);	
	
	cuda_rv = cudaMemcpy(batch_ce, &init_ce, sizeof(float), cudaMemcpyHostToDevice);  
	gpuAssert(cuda_rv);
	
	float* z2d; // derivative wrt z2 = y_predict - y_true
	gpu_malloc_size = NUM_LABELS * sizeof(float);
	cuda_rv = cudaMalloc(&z2d, gpu_malloc_size);
	gpuAssert(cuda_rv);

	float* w2d; // derivative wrt w2 = x2 * z2d
	gpu_malloc_size = NUM_LABELS * (HIDDEN_NEURONS + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&w2d, gpu_malloc_size);
	gpuAssert(cuda_rv);
				// initialize w2d to zero
	float w2d_initializer[NUM_LABELS * (HIDDEN_NEURONS + 1)] = { 0.0 };
	cuda_rv = cudaMemcpy(w2d, w2d_initializer, gpu_malloc_size, cudaMemcpyHostToDevice);
	gpuAssert(cuda_rv);
		
	float * x2d; // derivative wrt x2 = z2d * w2
	gpu_malloc_size = 1 * (HIDDEN_NEURONS + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&x2d, gpu_malloc_size);


	float* z1d; // derivatice wrt z1 = tensor product of x2d and derivative_relu( z1 )
	gpu_malloc_size = 1 * HIDDEN_NEURONS * sizeof(float);
	cuda_rv = cudaMalloc(&z1d, gpu_malloc_size);
	gpuAssert(cuda_rv);
	
	float * w1d; // derivative wrt w1 = x1 * z1d
	gpu_malloc_size = HIDDEN_NEURONS * (NUM_FEATURES + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&w1d, gpu_malloc_size);
	gpuAssert(cuda_rv);
				// initialize w1d to zero
	float w1d_initializer[HIDDEN_NEURONS * (NUM_FEATURES + 1)] = { 0.0 };
	cuda_rv = cudaMemcpy(w1d, w1d_initializer, gpu_malloc_size, cudaMemcpyHostToDevice);
	gpuAssert(cuda_rv);



	// shuffler holds indicies into the train set
	// the indicies are shuffled each epoch to randomize the set order
	std::vector<int> shuffler(NUM_TRAIN_INSTANCES);	
	std::iota(shuffler.begin(), shuffler.end(), 0);
	
	std::uniform_int_distribution<int> uniform_dist(0, 99999);
	std::random_device r;	
	#ifdef NRAND	
	std::default_random_engine eng(1); // constant seed for debugging
	#else	
	std::default_random_engine eng(r());
	#endif		
	
	int weight_rand = uniform_dist(eng);

	
	// initialize weights to random values in the range [-0.01, 0.01]-----------
	// w1 are the weights associated with transition input_layer -> hidden_layer	
	init_weights<<<5, 512>>>(w1, HIDDEN_NEURONS, (NUM_FEATURES + 1), weight_rand + time(NULL)); 
	syncPeek();							
	
	weight_rand = uniform_dist(eng);
	
	// w2 are the weights associated with transition hidden_layer -> output_layer
	init_weights<<<5, 512>>>(w2, NUM_LABELS, (HIDDEN_NEURONS + 1), weight_rand + time(NULL));	
	syncPeek();

	std::cout << "\nTraining model..........\n" << std::endl;
	
	size_t epoch;	
	size_t batch_begin;	
	size_t batch_end;
	size_t batch_idx;
	size_t batch_counter;
	size_t shuffler_idx;
	size_t instance_idx;	

	float * instance_ptr;
	float * x1_batch_alias;
	float * x1i_alias;
	
	unsigned char * instance_label_ptr;
	unsigned char * y_true_batch_alias;
	unsigned char * yi_true_alias;

	int randctr;
	START_TIMER(1);
	for(epoch = 0; epoch < EPOCHS; epoch++){
		std::cout << "EPOCH: " << (epoch + 1) << " out of " << EPOCHS << "\n" << std::endl;		
		std::shuffle(shuffler.begin(), shuffler.end(), eng);
		shuffler_idx = 0;		
		batch_counter = 0;		
		randctr = 0;		
		for(batch_begin = 0; batch_begin < NUM_TRAIN_INSTANCES; batch_begin += BATCH_SIZE){		
			x1_batch_alias = x1_batch;											// ... grab ptrs to batch input device memory ...	
			y_true_batch_alias = y_true_batch;			
			batch_end = batch_begin + BATCH_SIZE;			
			#ifdef PBATCH			
			std::cout << "BATCH: " << (batch_counter + 1) << " / " << NUM_BATCHES;
			std::cout << " .. EPOCH: " << (epoch + 1) << " / " << EPOCHS << std::endl;
			#endif
			for(batch_idx = batch_begin; batch_idx < batch_end; batch_idx++){ 	// ... gather batch of random instance permutation ...										
				instance_idx = shuffler.at(shuffler_idx);						// ... obtain random instance and add it to batch device memory	...	
				instance_ptr = &(train_feature_set[instance_idx][0]);			// ... get random instance from train set ...
			
				cuda_rv = cudaMemcpy(x1_batch_alias, instance_ptr, INSTANCE_BYTES, cudaMemcpyHostToDevice); 
				gpuAssert(cuda_rv);	
						
				instance_label_ptr = &(train_label_set[instance_idx][0]);		// ... obtain associated one-hot instance label array ...
				cuda_rv = cudaMemcpy(y_true_batch_alias, instance_label_ptr, NUM_LABELS, cudaMemcpyHostToDevice);
				gpuAssert(cuda_rv);

				x1_batch_alias += (NUM_FEATURES + 1);							// ... increment instance/label device memory ptrs ...
				y_true_batch_alias += NUM_LABELS;
				++shuffler_idx;
			}
			x1i_alias = x1_batch;												// ... reset device mem batch ptrs to beginning ...
			yi_true_alias = y_true_batch;
			
			for(int i =0; i < BATCH_SIZE; i++){ 								// ... train on batch ...
		
				//START_TIMER(2);
				multiplyX1W1<<<HIDDEN_NEURONS, 256>>>(x1i_alias, w1, z1, HIDDEN_NEURONS, (NUM_FEATURES + 1)); //XXX fn_id = 2
				//STOP_TIMER(2);
				syncPeek();


				//START_TIMER(4);	
				reLu_train<<<1, HIDDEN_NEURONS>>>(z1, x2, relud, time(NULL) + shuffler.at(randctr)); randctr++;
				//STOP_TIMER(4);
				syncPeek();
				
				//START_TIMER(5);	

				multiplyX2W2<<<NUM_LABELS, 512>>>(x2, w2, z2, NUM_LABELS, (HIDDEN_NEURONS + 1));  // XXX fn_id = 5			
				//STOP_TIMER(5);
				syncPeek();
		

				//START_TIMER(7);
				softmax<<<1, 10>>>(z2, y_soft_predict);	
				//STOP_TIMER(7);
				syncPeek();

				//START_TIMER(8);
				z2d_and_ce<<<1, 10>>>(y_soft_predict, yi_true_alias, z2d, epoch_ce, batch_ce, i, BATCH_SIZE, batch_counter, NUM_BATCHES);
				//STOP_TIMER(8);
				syncPeek();
				
				//START_TIMER(9);
				calcW2d<<<1, 1024>>>(x2, z2d, w2d, NUM_LABELS, (HIDDEN_NEURONS + 1));
				//STOP_TIMER(9);
				syncPeek();	
				
				//START_TIMER(10);
				calcz1d<<<1, 1024>>>(z2d, w2, z1d, relud, NUM_LABELS, (HIDDEN_NEURONS + 1));
				//STOP_TIMER(10);
				syncPeek();

				//START_TIMER(11);
				calcW1d<<<32, 1024>>>(x1i_alias, z1d, w1d, HIDDEN_NEURONS, (NUM_FEATURES+1));	
				//STOP_TIMER(11);
				syncPeek();

				x1i_alias += (NUM_FEATURES + 1);
				yi_true_alias += NUM_LABELS;
			}
			updateW1<<<32, 1024>>>(w1, w1d, BATCH_SIZE, LEARNING_RATE);
			syncPeek();
			updateW2<<<1, 1024>>>(w2, w2d, BATCH_SIZE, LEARNING_RATE);
			syncPeek();
			++batch_counter;
		}
	}
	printf("Model training duration:\n");
	STOP_TIMER(1);
	float* x1_test_set; // copy whole test set to device memory
	gpu_malloc_size = NUM_TEST_INSTANCES * (NUM_FEATURES + 1) * sizeof(float);
	cuda_rv = cudaMalloc(&x1_test_set, gpu_malloc_size);
	gpuAssert(cuda_rv);
	cuda_rv = cudaMemcpy(x1_test_set, &(test_feature_set[0][0]), gpu_malloc_size, cudaMemcpyHostToDevice); 
	gpuAssert(cuda_rv);	

	unsigned char* y_true_test_set; // copy all associated one-hot label vectors
	gpu_malloc_size = NUM_TEST_INSTANCES * NUM_LABELS * sizeof(unsigned char);
	cuda_rv = cudaMalloc(&y_true_test_set, gpu_malloc_size);
	gpuAssert(cuda_rv);
	cuda_rv = cudaMemcpy(y_true_test_set, &(test_label_set[0][0]), gpu_malloc_size, cudaMemcpyHostToDevice); 
	gpuAssert(cuda_rv);	

	int* num_correct; // keep track of number of correct classifications in device memory
	gpu_malloc_size = sizeof(int);
	cuda_rv = cudaMalloc(&num_correct, gpu_malloc_size);
	gpuAssert(cuda_rv);
	int num_correct_initializer = 0;	
	cuda_rv = cudaMemcpy(num_correct, &num_correct_initializer, gpu_malloc_size, cudaMemcpyHostToDevice);
	gpuAssert(cuda_rv);
	
	x1i_alias = x1_test_set; // set pointers to beginning of test data in device memory
	yi_true_alias = y_true_test_set;
	printf("\nTesting model..........\n");	
	for(int i = 0; i < NUM_TEST_INSTANCES; i++){
		multiplyX1W1<<<HIDDEN_NEURONS, 256>>>(x1i_alias, w1, z1, HIDDEN_NEURONS, (NUM_FEATURES + 1));		// XXX fn_id = 2
		syncPeek();							

		reLu<<<1, HIDDEN_NEURONS>>>(z1, x2, relud);
		syncPeek();

		multiplyX2W2<<<NUM_LABELS, 512>>>(x2, w2, z2, NUM_LABELS, (HIDDEN_NEURONS + 1));  // XXX fn_id = 5
		syncPeek();
			
		softmax<<<1, 10>>>(z2, y_soft_predict);
		syncPeek();
		
		test_run<<<1, 10>>>(y_soft_predict, yi_true_alias, num_correct);
		syncPeek();
		x1i_alias += (NUM_FEATURES + 1);
		yi_true_alias += NUM_LABELS;
	}		
	
	int result;
	
	cuda_rv = cudaMemcpy(&result, num_correct, sizeof(int), cudaMemcpyDeviceToHost);
	gpuAssert(cuda_rv);
	
	float percent = ((float)result) / ((float)NUM_TEST_INSTANCES);	
	
	std::cout << "\nModel Configuration:" << std::endl;	
	std::cout << "EPOCHS\t\tBATCH_SIZE\t\tLEARNING_RATE" << std::endl;
	std::cout << EPOCHS << "\t\t" << BATCH_SIZE << "\t\t\t" << LEARNING_RATE << std::endl;
	std::cout << "\nTest Result:\n" << std::endl;
	std::cout << "CORRECT: " << result << " / " << NUM_TEST_INSTANCES << std::endl;
	printf("ACCURACY: %f\n\n\n\n", percent);


	cuda_rv = cudaFree(x1_batch);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(w1);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(z1);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(relud);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(x2);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(w2);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(z2);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(y_soft_predict);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(y_true_batch);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(epoch_ce);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(batch_ce);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(z2d);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(w2d);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(x2d);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(z1d);
	gpuAssert(cuda_rv);	
	cuda_rv = cudaFree(w1d);
	gpuAssert(cuda_rv);


	cuda_rv = cudaFree(x1_test_set);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(y_true_test_set);
	gpuAssert(cuda_rv);
	cuda_rv = cudaFree(num_correct);
	gpuAssert(cuda_rv);
	
	return 0;
}



void get_hyper_parameters(float& LEARNING_RATE, long& EPOCHS, long& BATCH_SIZE, long& NUM_BATCHES, char** argv){
	LEARNING_RATE = atof(argv[1]); assert(LEARNING_RATE > 0.0);
	EPOCHS = atol(argv[2]); assert(EPOCHS > 0);
	BATCH_SIZE = atol(argv[3]); assert(BATCH_SIZE > 0);	
	
	if(NUM_TRAIN_INSTANCES % BATCH_SIZE){
		std::cout << "BATCH_SIZE must be a factor of NUM_TRAIN_INSTANCES\n" << std::endl;
		std::cout << BATCH_SIZE << " is not a factor of " << NUM_TRAIN_INSTANCES << "\n" << std::endl;
		exit(0);
	}
	NUM_BATCHES = NUM_TRAIN_INSTANCES / BATCH_SIZE;

	std::cout << "EPOCHS:\t\t" << EPOCHS << std::endl;
	std::cout << "BATCH_SIZE:\t" << BATCH_SIZE  << std::endl;
	std::cout << "LEARNING_RATE:\t" << LEARNING_RATE << std::endl;
}



//------------------------------------------------------------------------------
// sandbox experiments to understand cuda thread indexing-----------------------
void test_wrapper(){
	int BLOCKX = 32;
	int	BLOCKY = 16;

	int DIMX = 10;
	int DIMY = 20;
	
	dim3 block(BLOCKX ,BLOCKY);
	
	int gridx = (DIMX+block.x-1)/block.x;
	int gridy = (DIMY+block.y-1)/block.y;
		
	dim3 grid(gridx, gridy);	
	
	int num_test_threads = BLOCKX * BLOCKY * gridx * gridy;
	std::cout << "test<<<>>>() \n" << std::endl;

	test<<<grid, block>>>();
	syncPeek();	

	
	std::cout << "num_test_threads: " << num_test_threads << "\n"  << std::endl;
	std::cout << "END test<<<>>>() \n" << std::endl;
}

void index_test_wrapper(){ //XXX this is not finished 
	
	int g = 3;
	int b = 4;	
	
	dim3 grid1(g);
	dim3 grid2(g);
	dim3 grid3(g);	
	
	dim3 block1(b);
	dim3 block2(b, b);
	dim3 block3(b, b, b);
	
	dim3 grid4(g, g);
	dim3 grid5(g, g);
	dim3 grid6(g, g);
	dim3 grid7(g, g);
	dim3 grid8(g, g);
	dim3 grid9(g, g);		
	
	dim3 block4(b);
	dim3 block5(b);
	dim3 block6(b, b);
	dim3 block7(b, b);
	dim3 block8(b, b, b);
	dim3 block9(b, b, b);
	
	dim3 grid10(g, g, g);
	dim3 grid11(g, g, g);
	dim3 grid12(g, g, g);
	dim3 grid13(g, g, g);
	dim3 grid14(g, g, g);
	dim3 grid15(g, g, g);
	
	dim3 block10(b);
	dim3 block11(b);
	dim3 block12(b, b);
	dim3 block13(b, b);
	dim3 block14(b, b, b);
	dim3 block15(b, b, b);
	
	
	std::cout << "index_test<<<>>>() \n" << std::endl;
	
	std::cout << ".......... grid1(" << g << ")\tblock1(" << b << ")" << std::endl;
	indexing_test<<<grid1, block1>>>();
	syncPeek();
	std::cout << ".......... grid2(" << g << ")\tblock1(" << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid2, block2>>>();
	syncPeek();
	std::cout << ".......... grid3(" << g << ")\tblock1(" << b << ", " << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid3, block3>>>();
	syncPeek();
	std::cout << ".......... grid4(" << g << ", " << g << ")\tblock1(" << b << ")" << std::endl;
	indexing_test<<<grid4, block4>>>();
	syncPeek();
	std::cout << ".......... grid5(" << g << ", " << g << ")\tblock1(" << b << ")" << std::endl;
	indexing_test<<<grid5, block5>>>();
	syncPeek();
	std::cout << ".......... grid6(" << g << ", " << g << ")\tblock1(" << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid6, block6>>>();
	syncPeek();
	std::cout << ".......... grid7(" << g << ", " << g << ")\tblock1(" << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid7, block7>>>();
	syncPeek();
	std::cout << ".......... grid8(" << g << ", " << g << ")\tblock1(" << b << ", " << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid8, block8>>>();
	syncPeek();
	std::cout << ".......... grid9(" << g << ", " << g << ")\tblock1(" << b << ", " << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid9, block9>>>();
	syncPeek();
	std::cout << ".......... grid10(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ")" << std::endl;
	indexing_test<<<grid10, block10>>>();
	syncPeek();
	std::cout << ".......... grid11(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ")" << std::endl;
	indexing_test<<<grid11, block11>>>();
	syncPeek();
	std::cout << ".......... grid12(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid12, block12>>>();
	syncPeek();
	std::cout << ".......... grid13(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid13, block13>>>();
	syncPeek();
	std::cout << ".......... grid14(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ", " << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid14, block14>>>();
	syncPeek();
	std::cout << ".......... grid15(" << g << ", " << g << ", " << g << ")\tblock1(" << b << ", " << b << ", " << b << ")" << std::endl;
	indexing_test<<<grid15, block15>>>();
	syncPeek();

}



