#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void
k_vector_add(const float* A, const float* B, float* C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		C[i] = A[i] + B[i];
}

__global__ void
k_dot(float* a, float* b, float* c, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) {
		c[index] = a[index] * b[index];
	}
}

float *dot(float* a, float* b, int size){
	float *c;
	gpuErrchk(cudaMalloc(&c, size * sizeof(float)));
	k_dot <<<1,size>>> (a,b,c,size);
	return c;
}

__global__ void
k_scale(float scalar, float* vector, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) {
		vector[index] = vector[index] * scalar;
	}
}

void scale(float scalar, float* vector, int size){
	k_scale <<<1,size>>> (scalar,vector,size);
}