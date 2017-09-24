/*
 ============================================================================
 Name        : testers.cu
 Author      : BlazingDB
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <ctime>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "BlazingCachedAllocator.cuh"

template <typename T> __host__ __device__  T reciprocal(const T &x)
{
	return 2 * x;
}

template <typename T> class ReciprocalFunctor {
	public:
	__host__ __device__ T operator()(const T &x) {
		return reciprocal(x);
	}
};

template <typename T, class OpClass> T transformAndSumCPU(std::vector<T> data, OpClass op)
{
	std::vector<T> temp(data.size());
	std::transform(data.begin(), data.end(), temp.begin(), op);
	return std::accumulate(temp.begin(), temp.end(), (T)0);
}

template <typename T, class OpClass> T transformAndSumGPU(std::vector<T> data, OpClass op)
{

	std::clock_t begin = std::clock();
		thrust::device_vector<T> temp( data.end() - data.begin());
		std::clock_t end = std::clock();

		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout<<"Seconds Allocating Slow"<<elapsed_secs<<std::endl;

		begin = std::clock();
			thrust::copy( data.begin(), data.end(),temp.begin());
			end = std::clock();

			elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout<<"Seconds Copying Slow"<<elapsed_secs<<std::endl;

		begin = std::clock();
		thrust::transform(temp.begin(), temp.end(), temp.begin(), op);
		end = std::clock();

		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout<<"Seonds  transforming Slow"<<elapsed_secs<<std::endl;


		begin = std::clock();
			T result = thrust::reduce(temp.begin(), temp.end());
				end = std::clock();

				elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
				std::cout<<"Seonds  Reducing Slow"<<elapsed_secs<<std::endl;


			return result;
}


template <typename T, class OpClass> T transformAndSumGPUCustomTempAlloc(std::vector<T> data, OpClass op)
{

	std::clock_t begin = std::clock();
	thrust::device_vector<T, BlazingDeviceAllocator<T> > temp( data.end() - data.begin());
	std::clock_t end = std::clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"Seconds Allocating"<<elapsed_secs<<std::endl;

	begin = std::clock();
		thrust::copy( data.begin(), data.end(),temp.begin());
		end = std::clock();

		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout<<"Seconds Copying"<<elapsed_secs<<std::endl;

	begin = std::clock();
	thrust::transform(thrust::cuda::par(cachedDeviceAllocator),temp.begin(), temp.end(), temp.begin(), op);
	end = std::clock();

	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"Seonds  transforming"<<elapsed_secs<<std::endl;
	begin = std::clock();
	T result = thrust::reduce(thrust::cuda::par(cachedDeviceAllocator),temp.begin(), temp.end());
		end = std::clock();

		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout<<"Seonds  Reducing"<<elapsed_secs<<std::endl;


	return result;
}


template<typename T> void initialize(std::vector<T> &data, unsigned workSize)
{
	/* Initialize the vector */
	for (unsigned i = 0; i < workSize; i++)
		data.push_back( ((T)0.1)*(i+1) );
}
int iterations =  5;
template<typename T> void doCompute(unsigned workSize)
{
	std::vector<T> hostData;

	initialize(hostData, workSize);
	T cpuResults = transformAndSumCPU(hostData, ReciprocalFunctor<T>());

	std::clock_t begin = std::clock();
	T gpuResults;
	for(int i = 0; i < iterations; i++){
		 gpuResults += transformAndSumGPU(hostData, ReciprocalFunctor<T>());
		

	}

	std::clock_t end = std::clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	begin = std::clock();

	for(int i = 0; i < iterations; i++){
			gpuResults += transformAndSumGPUCustomTempAlloc(hostData, ReciprocalFunctor<T>());

	}

	end = std::clock();

	double elapsed_secs_custom = double(end - begin) / CLOCKS_PER_SEC;


	std::cout<<"transformAndSumCPU = "<<cpuResults<<std::endl;
	std::cout<<"transformAndSumGPU = "<<gpuResults<<" in thiis many seconds "<<elapsed_secs<<std::endl;
	std::cout<<"transformAndSumGPUCustomTempAlloc = "<<gpuResults<<" in thiis many seconds "<<elapsed_secs_custom<<std::endl;
}



int main(void)
{
	int * x;
	cudaMalloc((void **) &x, 100000);
	cudaFree(x);

	doCompute<long long> (1024*1024*200);
	return 0;
}
