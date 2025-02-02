
#include "cudaLib.cuh"
#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < size){
        y[i] = scale*x[i] + y[i];
    }
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
    float *a = (float *) malloc(vectorSize * sizeof(float));
    float *b = (float *) malloc(vectorSize * sizeof(float));
    float *c = (float *) malloc(vectorSize * sizeof(float));

    if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}
    
    vectorInit(a, vectorSize);
    vectorInit(b, vectorSize);

    float *a_d, *b_d, *c_d;
    float scale = (float)(rand() % 100);    //random scaler

    // Calculate threadBlock
    int threadPerBlock = 256;
    int nblocks = (vectorSize + threadPerBlock - 1)/threadPerBlock;

    // allocate a_d and b_d for x and y on GPU and copy from CPU
    cudaMalloc(&a_d, vectorSize*sizeof(float));
    cudaMalloc(&b_d, vectorSize*sizeof(float));
    cudaMemcpy(a_d, a, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&c_d, vectorSize*sizeof(float));
    cudaMemcpy(c_d, b, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
    #ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif
    // Kernel
    saxpy_gpu<<<nblocks,threadPerBlock>>>(a_d, c_d, scale, vectorSize);

    // Transfer result back to CPU
    cudaMemcpy(c, c_d, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);

    #ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif
    int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

    // Free Device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    // Free host memory
    free(a);
    free(b);
    free(c);
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    curandState_t rng;
	curand_init(clock64(), i, 0, &rng);
    float x, y;
    uint64_t hitCount = 0;

    // pSumSize of threads
    if(i < pSumSize){
        // calculate the hit count on each thread (size: sampleSize)
        for (uint64_t idx = 0; idx < sampleSize; idx ++){
            x = curand_uniform(&rng);
            y = curand_uniform(&rng);
            if ((x * x + y * y) <= 1.0f) {
                ++ hitCount;
            }
        }
        // put the hitCount in array pSums
        pSums[i] = hitCount;
    }
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    uint64_t totalCount = 0;
    uint64_t size = pSumSize/reduceSize;
    if(i < size){
        for (uint64_t j = 0; j < reduceSize; j ++){
            uint64_t idx = i*reduceSize + j;
            if (idx < pSumSize) {
                totalCount += pSums[idx];
            }
        }
        totals[i] = totalCount;
    }
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

    // Insert code here
    uint64_t *pSums_d;
    uint64_t *totals_d;
    uint64_t totalHitCount = 0;

    // malloc pSums_d and totals_d on GPU
	cudaMalloc(&pSums_d, generateThreadCount*sizeof(uint64_t));
    cudaMalloc(&totals_d, reduceThreadCount*sizeof(uint64_t));

    // calculate the block number
    int threadPerBlock = 256;
    int numBlocks = (generateThreadCount + threadPerBlock - 1)/threadPerBlock;
    int numBlocks_reduce = (reduceThreadCount + threadPerBlock - 1)/threadPerBlock;

    // Get pSum_d from generatePoint and caculate the hitcount on GPU
    generatePoints<<<numBlocks, threadPerBlock>>>(pSums_d, generateThreadCount, sampleSize);

    // Generate totals_d, take pSums_d to reduce count on GPU
    reduceCounts<<<numBlocks_reduce, threadPerBlock>>>(pSums_d, totals_d, generateThreadCount, reduceSize);

    // Malloc total_h on CPU and copy the reduce hit count: totals_d to CPU
    uint64_t *total_h = (uint64_t *) malloc(reduceThreadCount*sizeof(uint64_t));
    cudaMemcpy(total_h, totals_d, reduceThreadCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // for loop on CPU to calculate totalHitCount
    for(uint64_t idx = 0; idx < reduceThreadCount; idx++){
        totalHitCount += total_h[idx];
    }

    // final calculation on pi
	approxPi = ((double)totalHitCount/sampleSize)/generateThreadCount;
    approxPi = approxPi*4;
    
    // Free Device memory
    cudaFree(pSums_d);
    cudaFree(totals_d);
    // Free Host memory
    free(total_h);
	return approxPi;
}
