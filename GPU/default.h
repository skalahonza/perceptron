#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include <device_functions.h>

/**
 * @brief Assert CUDA operation call
 * 
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}