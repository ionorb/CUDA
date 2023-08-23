#include <cuda_runtime.h>
#include <iostream>

// #include "cuda_helper.h"

int	main(void)
{
	int	count;

	cudaGetDeviceCount(&count);
	std::cout << "count: " << count << std::endl;
	return (0);
}