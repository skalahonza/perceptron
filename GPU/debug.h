#pragma once
#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#include "cuda_runtime.h"
using namespace std;

/**
 * @brief Print cuda array into console
 * 
 * @param data Cuda array data
 * @param size Size of array
 */
void print_cuda_array(int* data, int size)
{
	int* tmp = (int*)calloc(size, sizeof(int));
	cudaMemcpy(tmp, data, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (size_t j = 0; j < size; j++)
	{
		cout << tmp[j] << " ";
	}
	cout << endl;
	free(tmp);
}

/**
 * @brief Print cuda float array
 * 
  * @param data Cuda array data
 * @param size Size of array
 */
void print_cuda_array(float* data, int size)
{
	float* tmp = (float*)calloc(size, sizeof(float));
	cudaMemcpy(tmp, data, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t j = 0; j < size; j++)
	{
		cout << tmp[j] << " ";
	}
	cout << endl;
	free(tmp);
}

/**
 * @brief Print cuda 2D array (matrix) into console
 * 
 * @param data Cuda 2D array data
 * @param width Array width
 * @param height Array height
 */
void print_cuda2d_array(float* data, int width, int height)
{
	size_t size = width * height;
	float* tmp = (float*)calloc(size, sizeof(float));
	cudaMemcpy(tmp, data, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			cout << tmp[i * (size_t)width + j] << " ";
		}
		cout << endl;
	}
	free(tmp);
}

#endif