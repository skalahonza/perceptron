#include "kernels.cuh"
#define min(x,y) (x>y?x:y)
#define ThreadPerBlock 256
//smallest multiple of threadsPerBlock
#define blockPerGrid(n) min(32, ((n)+ThreadPerBlock-1) / ThreadPerBlock)

__global__ void
k_vector_add(float* A, const float* B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		A[i] = A[i] + B[i];
}

void vector_add(float* A, const float* B, int numElements) {
	k_vector_add << <1, numElements >> > (A, B, numElements);
}

__device__ void
d_dot(const float* V1, const float* V2, float* V3, int size)
{
	__shared__ float chache[ThreadPerBlock];
	float temp;

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int chacheindex = threadIdx.x;

	while (tid < size)
	{
		temp += V1[tid] * V2[tid];
		tid += blockDim.x * gridDim.x;
	}

	chache[chacheindex] = temp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (chacheindex < i)
			chache[chacheindex] += chache[chacheindex + i];
		__syncthreads();
		i >>= 1;
	}

	if (chacheindex == 0)
		V3[blockIdx.x] = chache[0];
}

__global__ void
k_dot(const float* V1, const float* V2, float* V3, int size)
{
	d_dot(V1, V2, V3, size);
}

__global__ void
k_update(float learn_rate, float* expected, float* data, float bias, float* weights, int size, float* result)
{
	d_dot(data, weights, result, size);
	*result += bias;
	*result *= learn_rate;
}

float* dot(float* a, float* b, int size) {
	float* c;
	gpuErrchk(cudaMalloc(&c, 1 * sizeof(float)));
	k_dot << <blockPerGrid(size), ThreadPerBlock >> > (a, b, c, size);
	return c;
}

float* update(float learn_rate, float* expected, float* data, float bias, float* weights, int size)
{
	float* result;
	gpuErrchk(cudaMalloc(&result, 1 * sizeof(float)));
	k_update << <blockPerGrid(size), ThreadPerBlock >> > (learn_rate, expected, data, bias, weights, size, result);
	return result;
}

__global__ void
k_scale(float *scaler, float* vector, float *result, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float s = *scaler;
	if (index < size) {
		result[index] += vector[index] * s;
	}
}

void scale(float *scaler, float* vector, float *result, int size) {
	k_scale << <1, size >> > (scaler, vector, result, size);
}