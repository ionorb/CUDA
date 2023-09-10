#include <cuda_runtime.h>
#include <iostream>

#define VEC_SIZE 25
#define NUM_THREADS 256
#define NUM_BLOCKS 1//(int)ceil(VEC_SIZE / NUM_THREADS)
// #include "cuda_helper.h"


void	vecAddOneHost(int*	host_vector, int*	host_result, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		host_result[i] = host_vector[i] + 1;
	}
}

__global__ void	vecAddOneDevice(int* device_vector, int* device_result, int size)
{
	unsigned int	i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < VEC_SIZE)
		device_result[i] = device_vector[i] + 1;
}

int	main(void)
{
	try
	{
		cudaError_t	err = cudaSuccess;

		int	*device_vector = NULL;
		int	*host_vector = NULL;

		int	*device_result = NULL;
		int	*host_result = NULL;

		int	*local_device_result = NULL;

		host_vector = (int *)malloc(sizeof(int) * VEC_SIZE);
		host_result = (int *)malloc(sizeof(int) * VEC_SIZE);
		local_device_result = (int *)malloc(sizeof(int) * VEC_SIZE);

		for (size_t i = 0; i < VEC_SIZE; i++)
			host_vector[i] = i;

		err = cudaMalloc(&device_vector, sizeof(int) * VEC_SIZE);
		if (err != cudaSuccess)
			throw std::runtime_error("error mallocing device_vector: " + std::string(cudaGetErrorString(err)));
		err = cudaMalloc(&device_result, sizeof(int) * VEC_SIZE);
		if (err != cudaSuccess)
			throw std::runtime_error("error mallocing device_result: " + std::string(cudaGetErrorString(err)));
		err = cudaMemcpy(device_vector, host_vector, sizeof(int) * VEC_SIZE, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			throw std::runtime_error("error cudaMemcpy host to device: " + std::string(cudaGetErrorString(err)));
	
		vecAddOneHost(host_vector, host_result, VEC_SIZE);
		std::cout << "calling vecAddOneDevice with NUM_BLOCKS = " << NUM_BLOCKS << ", NUM_THREADS = " << NUM_THREADS << std::endl;
		vecAddOneDevice<<<NUM_BLOCKS, NUM_THREADS>>>(device_vector, device_result, VEC_SIZE);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error("error after vecAddOneDevice(): " + std::string(cudaGetErrorString(err)));
		}
		err = cudaMemcpy(local_device_result, device_result, sizeof(int) * VEC_SIZE, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
			throw std::runtime_error("error cudaMemcpy device to host: " + std::string(cudaGetErrorString(err)));

		std::cout << "\nhost: " << std::endl;
		for (size_t i = 0; i < VEC_SIZE; i++)
			std::cout << host_result[i] << ", ";
		std::cout << "end" << std::endl;

		std::cout << "\ndevice: " << std::endl;
		for (size_t i = 0; i < VEC_SIZE; i++)
			std::cout << local_device_result[i] << ", ";
		std::cout << "end" << std::endl;

		free(host_vector);
		free(host_result);
		free(local_device_result);

		cudaFree(device_vector);
		cudaFree(device_result);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	return (0);
}