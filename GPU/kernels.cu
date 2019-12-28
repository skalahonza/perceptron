#include "kernels.cuh"
#define THREADS_PER_BLOCK 256

__global__ void
k_vector_add(float* A, const float* B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		A[i] = A[i] + B[i];
}

/// Add two vectors
void vector_add(float* A, const float* B, int numElements) {
	k_vector_add << <1, numElements >> > (A, B, numElements);
}

__device__ void
d_dot(const float* v1, const float* v2, float* out, int size)
{
	__shared__ float cache[THREADS_PER_BLOCK];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	cache[threadIdx.x] = 0.f;
	while (i < size) {
		cache[threadIdx.x] += v1[i] * v2[i];
		i += gridDim.x * blockDim.x;
	}
	__syncthreads();
	i = THREADS_PER_BLOCK / 2;
	while (i > 0) {
		if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0) atomicAdd(out, cache[0]);
}

__global__ void
k_dot(const float* V1, const float* V2, float* V3, int size)
{
	d_dot(V1, V2, V3, size);
}

__global__ void
k_update(float learn_rate, float* expected, float* data, float* bias, float* weights, int size, float* result)
{
	d_dot(data, weights, result, size);
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		*result = (*bias + *result) > 0 ? 1.f : -1.f;
		*result = learn_rate * (*expected - *result);
	}
}

/// Compute dot product of two vectors
float* dot(float* a, float* b, int size) {
	float* c;
	gpuErrchk(cudaMalloc(&c, 1 * sizeof(float)));
	k_dot << <blockPerGrid(size), THREADS_PER_BLOCK >> > (a, b, c, size);
	return c;
}

/// Compute udpate value for training
float* update(float learn_rate, float* expected, float* data, float* bias, float* weights, int size)
{
	float* result;
	gpuErrchk(cudaMalloc(&result, 1 * sizeof(float)));
	int block_count = size / THREADS_PER_BLOCK
		+ ((size % THREADS_PER_BLOCK) ? 1 : 0); // alignment
	k_update << <block_count, THREADS_PER_BLOCK >> > (learn_rate, expected, data, bias, weights, size, result);
	return result;
}

__global__ void
k_scale(float* scaler, float* vector, float* result, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float s = *scaler;
	if (index < size) {
		result[index] += vector[index] * s;
	}
}

/// Scale vector with a given scaler and save to result
void scale(float* scaler, float* vector, float* result, int size) {
	k_scale << <1, size >> > (scaler, vector, result, size);
}