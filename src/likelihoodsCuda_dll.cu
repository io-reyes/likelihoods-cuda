#include <math.h>
#include <float.h>

#define THREADS_PER_BLOCK 64
#define BLOCKS_PER_GRID 65000
#define PI 3.14159265358979323846f
#define NUM_DIMS 3

// kernel
__global__ void likelihoods_kernel(float *pts, float *objectPoint, float *objectArea, 
								   float *likes,
								   int numPoints, int numSurfaces, 
								   float noise, float totalArea,
								   float cValue ) 
{
	///Calculate Point Index
	int pointIndex = (gridDim.x * blockIdx.x) + blockIdx.y;
	//int pointIndex = blockIdx.x;
	if(pointIndex < numPoints){
		int thread_id = threadIdx.x;
		
		__shared__ float partialReduc[THREADS_PER_BLOCK];
		
		// Read point data
		__shared__ float point[NUM_DIMS];
		if(thread_id < NUM_DIMS){
            point[thread_id] = pts[pointIndex * NUM_DIMS + thread_id];
		}
		__syncthreads();
			
		// Calculate pointLike for all surfaces
		int numLoops = (numSurfaces + blockDim.x - 1) / blockDim.x;
		int surfacePoint = thread_id;
		int i;
		float difX, difY, difZ, dist, n;
		partialReduc[thread_id] = FLT_MAX;
		for(i = 0;i < numLoops;i++){
			if( surfacePoint < numSurfaces ){
				difX = (point[0] - objectPoint[surfacePoint * NUM_DIMS + 0]); 
				difY = (point[1] - objectPoint[surfacePoint * NUM_DIMS + 1]);
				difZ = (point[2] - objectPoint[surfacePoint * NUM_DIMS + 2]); 
				dist = ((difX*difX) + (difY*difY) + (difZ*difZ));
				n = dist * (1.0f/(2 * (noise*noise)));
				// Partial Min reduction into shared memory
				if(n < partialReduc[thread_id]){
					partialReduc[thread_id] = n;
				}
			}
			surfacePoint += blockDim.x;
		}
		
		__syncthreads();
		
		// Min reduce partial reduction in shared memory
		int th;
		for(th = blockDim.x / 2; th > 0; th/=2){
			if(thread_id < th){
				if(partialReduc[thread_id + th] < partialReduc[thread_id]){
					partialReduc[thread_id] = partialReduc[thread_id + th];
				}
			}
			__syncthreads();
		}
		
		// Save Surface match
		float surfaceMatch = partialReduc[0];
		
		__syncthreads();
		
		// Partial Reduction on likelihood
		surfacePoint = thread_id;
		float objArea, pLike;
		partialReduc[thread_id] = 0.0f;
		for(i = 0;i < numLoops;i++){
			if( surfacePoint < numSurfaces ){
				difX = (point[0] - objectPoint[surfacePoint * NUM_DIMS + 0]); 
				difY = (point[1] - objectPoint[surfacePoint * NUM_DIMS + 1]);
				difZ = (point[2] - objectPoint[surfacePoint * NUM_DIMS + 2]); 
				dist = ((difX*difX) + (difY*difY) + (difZ*difZ));
				pLike = dist * (1.0f/(2 * (noise*noise)));
				
				objArea = objectArea[surfacePoint];
				partialReduc[thread_id] += (objArea / (cValue * totalArea)) * expf(-pLike + surfaceMatch);
			}
			surfacePoint += blockDim.x;
		}
		
		__syncthreads();
		
		// Sum reduce partial reduction in shared memory
		for(th = blockDim.x / 2; th > 0; th/=2){
			if(thread_id < th){
				partialReduc[thread_id] += partialReduc[thread_id + th];
			}
			__syncthreads();
		}
		
		// Thread 0 stores the result
		if(thread_id == 0){
			likes[pointIndex] = log(partialReduc[thread_id]) - surfaceMatch;
		}
	}
}


extern "C" __declspec(dllexport) float* likelihoodsCuda(double* pts_mat, int numPoints, float* objectPoint, int numSurfaces, float* objectArea, float noise){
	// Convert to single percision array
	float* pts = (float*)malloc(numPoints * NUM_DIMS * sizeof(float));
    float* obj_pts = (float*)malloc(numSurfaces * NUM_DIMS * sizeof(float));

    // flatten out the points tables so that the coordinate components are adjacent (i.e., x1 y1 z1 x2 y2 z2 etc)
    for(int i = 0; i < numPoints; i++){
        pts[(3 * i) + 0] = (float)pts_mat[(numPoints * 0) + i];
        pts[(3 * i) + 1] = (float)pts_mat[(numPoints * 1) + i];
        pts[(3 * i) + 2] = (float)pts_mat[(numPoints * 2) + i];
    }

    for(int i = 0; i < numSurfaces; i++){
        obj_pts[(3 * i) + 0] = (float)objectPoint[(numSurfaces * 0) + i];
        obj_pts[(3 * i) + 1] = (float)objectPoint[(numSurfaces * 1) + i];
        obj_pts[(3 * i) + 2] = (float)objectPoint[(numSurfaces * 2) + i];
    }
        
	
	// Initialize output vector
	float* likes = (float*)malloc(numPoints * sizeof(float));
	
	// Find total surface area (Could imp using thrust?)
	float totalArea = 0;
	for (int i = 0; i < numSurfaces; i++){
		totalArea += objectArea[i];
	}
	
	// Allocate Data on GPU
	float *pts_GPU, *objectPoint_GPU, *objectArea_GPU, *likes_GPU;
	cudaMalloc((void **) &pts_GPU, numPoints * NUM_DIMS * sizeof(float));
	cudaMalloc((void **) &objectPoint_GPU, numSurfaces * NUM_DIMS * sizeof(float));
	cudaMalloc((void **) &objectArea_GPU, numSurfaces * sizeof(float));
	cudaMalloc((void **) &likes_GPU, numPoints * sizeof(float));
	
	// Copy data to GPU
	cudaMemcpy(pts_GPU, pts, numPoints * NUM_DIMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(objectPoint_GPU, obj_pts, numSurfaces * NUM_DIMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(objectArea_GPU, objectArea, numSurfaces * sizeof(float), cudaMemcpyHostToDevice);
	
	///kernel Constants: numPoints,numSurfaces,noise,totalArea
	
	// Launch kernel
	int num_blocks = numPoints;
	int num_grids = (num_blocks + BLOCKS_PER_GRID - 1) / BLOCKS_PER_GRID;
	dim3 grid(num_grids, BLOCKS_PER_GRID, 1);
	dim3 thread_block(THREADS_PER_BLOCK, 1, 1);
	
	float cValue = pow(2 * PI * (noise*noise), 1.5f);
	
	likelihoods_kernel <<< grid, thread_block >>> (pts_GPU, 
												   objectPoint_GPU, 
												   objectArea_GPU, 
												   likes_GPU,
												   numPoints,
												   numSurfaces,
												   noise,
												   totalArea,
												   cValue );
	
	// Copy data back
	cudaMemcpy(likes, likes_GPU, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Free GPU Data
	cudaFree(pts_GPU);
	cudaFree(objectPoint_GPU);
	cudaFree(objectArea_GPU);
	cudaFree(likes_GPU);
	
	// Free CPU data
	free(pts);
    free(obj_pts);
	
	// Done
	return likes;
}
