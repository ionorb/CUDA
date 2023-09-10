#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#define VEC_SIZE 1000000
#define NUM_THREADS 769
#define NUM_BLOCKS (int)ceil(VEC_SIZE / NUM_THREADS) + 1
#define N 2000
// #include "cuda_helper.h"


void	vecAddOneHost(int*	host_vector, int*	host_result, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		host_result[i] = host_vector[i] + 1;
	}
}

__global__ void	vecAddOneDevice(int* device_vector, int* device_result, size_t size)
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

		int	*vector = NULL;
		int *host_result = NULL;
		int *device_result = NULL;

		err = cudaMallocManaged(&vector, sizeof(int) * VEC_SIZE);
		if (err != cudaSuccess)
			throw std::runtime_error("error allocating vector: " + std::string(cudaGetErrorString(err)));
		err = cudaMallocManaged(&host_result, sizeof(int) * VEC_SIZE);
		if (err != cudaSuccess)
			throw std::runtime_error("error allocating host_result: " + std::string(cudaGetErrorString(err)));
		err = cudaMallocManaged(&device_result, sizeof(int) * VEC_SIZE);
		if (err != cudaSuccess)
			throw std::runtime_error("error allocating device_result: " + std::string(cudaGetErrorString(err)));

		for (size_t i = 0; i < VEC_SIZE; i++)
			vector[i] = i;

		// std::cout << "\noriginal: " << std::endl;
		// for (size_t i = 0; i < VEC_SIZE; i++)
		// 	std::cout << vector[i] << ", ";
		// std::cout << "end" << std::endl;

		std::cout << "\ncalling vecAddOneHost...\n";
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < N; i++)
			vecAddOneHost(vector, host_result, VEC_SIZE);
		auto stop = std::chrono::high_resolution_clock::now();
		auto millisecs = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		double duration = (double)millisecs.count() / (double)1000;
		std::cout << "done in " << duration << " seconds\n";

		// std::cout << "\nhost: " << std::endl;
		// for (size_t i = 0; i < VEC_SIZE; i++)
		// 	std::cout << result[i] << ", ";
		// std::cout << "end" << std::endl;

		std::cout << "\ncalling vecAddOneDevice with NUM_BLOCKS = " << NUM_BLOCKS << ", NUM_THREADS = " << NUM_THREADS << "...\n";
		start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < N; i++)
			vecAddOneDevice<<<NUM_BLOCKS, NUM_THREADS>>>(vector, device_result, VEC_SIZE);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			throw std::runtime_error("error after vecAddOneDevice(): " + std::string(cudaGetErrorString(err)));
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
			throw std::runtime_error("error after synchronize: " + std::string(cudaGetErrorString(err)));
		stop = std::chrono::high_resolution_clock::now();
		millisecs = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		duration = (double)millisecs.count() / (double)1000;
		std::cout << "done in " << duration << " seconds\n";

		// std::cout << "\ndevice: " << std::endl;
		// for (size_t i = 0; i < VEC_SIZE; i++)
		// 	std::cout << result[i] << ", ";
		// std::cout << "end" << std::endl;

		for (size_t i = 0; i < VEC_SIZE; i++)
		{
			if (host_result[i] != device_result[i])
			{
				std::cerr << "host_result[" << i << "] = " << host_result[i] << "\ndevice_result[" << i << "] = " << device_result[i] << std::endl;
				throw std::runtime_error("results dont match!");
			}
		}
		std::cout << "results matched\n";

		cudaFree(vector);
		cudaFree(host_result);
		cudaFree(device_result);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	return (0);
}