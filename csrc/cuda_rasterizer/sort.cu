#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

void sort_gaussian(int num_rendered,
    int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted)
{
	cudaEvent_t start, stop;
	CHECK_CUDART(cudaEventCreate(&start));
	CHECK_CUDART(cudaEventCreate(&stop));
	CHECK_CUDART(cudaEventRecord(start, 0));  // Record on default stream

	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	auto status = cub::DeviceRadixSort::SortPairs(
		list_sorting_space, sorting_size,
		gaussian_keys_unsorted, gaussian_keys_sorted,
		gaussian_values_unsorted, gaussian_values_sorted,
		num_rendered, 0, 32 + getHigherMsb(grid.x * grid.y));
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }

	CHECK_CUDART(cudaEventRecord(stop, 0));  // Record on default stream
	CHECK_CUDART(cudaEventSynchronize(stop));  // CPU waits here until the kernel reaches this point
	float milliseconds = 0;
	CHECK_CUDART(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("sort: %f ms \n", milliseconds);
	CHECK_CUDART(cudaEventDestroy(start));
	CHECK_CUDART(cudaEventDestroy(stop));
}
