#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H


template<int NUM_INSTANCES>
void print_set(float (&feature_set)[NUM_INSTANCES][NUM_FEATURES+1], 
				unsigned char (&label_set)[NUM_INSTANCES][NUM_LABELS]){		
	
	int digit = -1;	
	for(int i = 0; i < NUM_INSTANCES; i++){		
		for(int z = 0; z < NUM_LABELS; z++){
			if(label_set[i][z]) {
				digit = z;
				break;
			}
		}
		std::cout << digit << std::endl;	
		for(int x = 0; x < IMAGE_DIM; x++){
			for(int y = 0; y < IMAGE_DIM; y++){
				if(feature_set[i][(x * IMAGE_DIM + y) + 1]){	
					std::cout << 0;			
				} else{

					std::cout << " ";
				}
				std::cout << " ";
			}
			std::cout << std::endl;
		}		
		std::cout << std::endl;
	}
}

void endian_switch(int & buf){
	int tmp = buf;
	unsigned char byte;	
	buf = 0;
	for(int i = 0, j = 24; i < 25; i += 8, j -= 8){
		byte = (tmp >> i) & 255;
		buf += ((int)byte << j);
	}
}

template<int NUM_INSTANCES>
void parse_features(const char* file_name, float (&feature_set)[NUM_INSTANCES][NUM_FEATURES + 1]){
	//std::cout << "Parsing features:\t" << file_name << std::endl;	
	int rv;	
	int buffer;
	unsigned char tmp_instance_features[NUM_FEATURES];	
	
	int fd = open(file_name, O_RDONLY);
	if(fd < 0){
		std::cout << "Error opening feature_set file: " << file_name << std::endl;
		exit(0);
	}
	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);
	assert(buffer == FEATURE_MAGIC);
	
	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);	
	assert(buffer == NUM_INSTANCES);

	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);	
	assert(buffer == IMAGE_DIM);
	
	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);	
	assert(buffer == IMAGE_DIM);
	
	for(int i = 0; i < NUM_INSTANCES; i++){
		rv = read(fd, &tmp_instance_features, NUM_FEATURES); assert(rv == NUM_FEATURES);
		feature_set[i][0] = 1.0;	
		for(int j = 1; j < NUM_FEATURES + 1; j++){
			feature_set[i][j] = (float)(tmp_instance_features[j - 1]);
			assert(feature_set[i][j] == tmp_instance_features[j - 1]);
		}

	}
	rv = close(fd);
	if(rv != 0){
		std::cout << "Error closing feature_set file: " << file_name << std::endl;
		exit(0);
	}
}  

template<int NUM_INSTANCES>
void parse_labels(const char* file_name, unsigned char (&label_set)[NUM_INSTANCES][NUM_LABELS]){
	//std::cout << "Parsing labels:\t\t" << file_name << std::endl;
	int rv;	
	int buffer;
	int fd = open(file_name, O_RDONLY);
	unsigned char tmp;
	if(fd < 0){
		std::cout << "Error opening label_set file: " << file_name << std::endl;
		exit(0);
	}
	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);
	assert(buffer == LABEL_MAGIC);

	rv = read(fd, &buffer, sizeof(int)); assert(rv == sizeof(int));
	endian_switch(buffer);	
	assert(buffer == NUM_INSTANCES);
	for(int i = 0; i < NUM_INSTANCES; i++){
		rv = read(fd, &tmp, sizeof(unsigned char)); assert(rv == sizeof(unsigned char));
		label_set[i][tmp] = 1;
	}
	rv = close(fd);
	if(rv != 0){
		std::cout << "Error closing label_set file:" << file_name << std::endl;
		exit(0);
	}
}  


#endif
