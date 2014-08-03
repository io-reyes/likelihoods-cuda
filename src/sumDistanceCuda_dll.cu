#include <math.h>
#include <float.h>

#define THREADS_PER_BLOCK 64
#define BLOCKS_PER_GRID 65000
#define PI 3.14159265358979323846f
#define NUM_DIMS 3

/*
  SUMDISTKERNEL  Parallel kernel for computing the sum of the distances between a point and all facet centerpoints
                 in the reference object.

  float* pts is a row-major flat array representing an N x 3 matrix of measured 3D points

  int numPts is the number of points in the above array

  float* objPts is a row-major flat array representing a K x 3 matrix of 3D facet center points

  int numObjPts is the number of points in the above array
  
  ---

  float* outDists is an N-length array into which the output point-to-all-surfaces distance will be written
 */

__global__ void sumDistKernel(float* pts, int numPts, float* objPts, int numObjPts, float* outDists) {
    // get the point # and initialize the output
	int ptIndex       = (gridDim.x * blockIdx.x) + blockIdx.y;
    outDists[ptIndex] = 0.0;

	if(ptIndex < numPts){
		int threadId = threadIdx.x;

        // read the point into shared memory
		__shared__ float pt[NUM_DIMS];
		if(threadId < NUM_DIMS){
            pt[threadId] = pts[ptIndex * NUM_DIMS + threadId];
		}
		__syncthreads();

        // sum up the distance between the point and all surfaces
        __shared__ float partialSum[THREADS_PER_BLOCK];
        partialSum[threadId] = 0.0;

        int surfacePt;
        float difX, difY, difZ;

        for(surfacePt = threadId; surfacePt < numObjPts; surfacePt += blockDim.x){
            difX = pt[0] - objPts[surfacePt * NUM_DIMS + 0];
            difY = pt[1] - objPts[surfacePt * NUM_DIMS + 1];
            difZ = pt[2] - objPts[surfacePt * NUM_DIMS + 2];

            partialSum[threadId] += sqrt((difX * difX) + (difY * difY) + (difZ * difZ));
        }
        __syncthreads();
		
		// Thread 0 stores the result
        int i;
        for(i = 0; threadId == 0 && i < THREADS_PER_BLOCK; i++)
            outDists[ptIndex] += partialSum[i];
	}
}

/*
  SUMDISTANCECUDA computes the sum of all point-to-surface distances.

  double* pts is a column-major flat array representing an N x 3 matrix of measured 3D points (as read in from MATLAB)
 
  int numPts is the number of points in the above array

  float* objPts is a column-major flat array representing a K x 3 matrix of 3D facet center points (as read in from MATLAB)

  int numObjPts is the number of points in the above array

  ---

  returns: the sum of point-to-surface distances
 */
extern "C" __declspec(dllexport) float sumDistanceCuda(double* pts, int numPts, float* objPts, int numObjPts){
    // declare loop variable
    int i;

	// allocate memory on the CPU
	float* rowPts    = (float*)malloc(numPts * NUM_DIMS * sizeof(float));
    float* rowObjPts = (float*)malloc(numObjPts * NUM_DIMS * sizeof(float));

    // flatten out the points tables so that the coordinate components are adjacent (i.e., x1 y1 z1 x2 y2 z2 etc)
    for(i = 0; i < numPts; i++){
        rowPts[(3 * i) + 0] = (float)pts[(numPts * 0) + i];
        rowPts[(3 * i) + 1] = (float)pts[(numPts * 1) + i];
        rowPts[(3 * i) + 2] = (float)pts[(numPts * 2) + i];
    }

    for(i = 0; i < numObjPts; i++){
        rowObjPts[(3 * i) + 0] = (float)objPts[(numObjPts * 0) + i];
        rowObjPts[(3 * i) + 1] = (float)objPts[(numObjPts * 1) + i];
        rowObjPts[(3 * i) + 2] = (float)objPts[(numObjPts * 2) + i];
    }

    // allocate memory on the GPU
    float *gpuPts, *gpuObjPts, *gpuDists;
    cudaMalloc((void**) &gpuPts   , NUM_DIMS * numPts    * sizeof(float));
    cudaMalloc((void**) &gpuObjPts, NUM_DIMS * numObjPts * sizeof(float));
    cudaMalloc((void**) &gpuDists ,            numPts    * sizeof(float));

    // copy data to GPU
    cudaMemcpy(gpuPts   , rowPts   , NUM_DIMS * numPts    * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuObjPts, rowObjPts, NUM_DIMS * numObjPts * sizeof(float), cudaMemcpyHostToDevice);

    // configure threads and launch kernel
    int numBlocks = numPts;
    int numGrids  = (numBlocks + BLOCKS_PER_GRID - 1) / BLOCKS_PER_GRID;

    dim3 grid(numGrids, BLOCKS_PER_GRID, 1);
    dim3 threadBlock(THREADS_PER_BLOCK, 1, 1);

    sumDistKernel <<< grid, threadBlock >>> (gpuPts, numPts, gpuObjPts, numObjPts, gpuDists);

    // copy the result and sum it up
    float* dists = (float*)malloc(numPts * sizeof(float));
    cudaMemcpy(dists, gpuDists,   numPts * sizeof(float), cudaMemcpyDeviceToHost);

    float sumDists = 0;
    for(i = 0; i < numPts; i++)
        sumDists += dists[i];

    // free memory on the GPU
    cudaFree(gpuPts);
    cudaFree(gpuObjPts);
    cudaFree(gpuDists);

    // free memory on the CPU
    free(rowPts);
    free(rowObjPts);
    free(dists);

    // return the result
    return sumDists;
}
