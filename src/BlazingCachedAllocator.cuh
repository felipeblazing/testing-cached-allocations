// this is taken from Thrust example at https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu

#ifndef BLAZINGCACHEALLOCATOR_H_
#define BLAZINGCACHEALLOCATOR_H_

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <cassert>
#include <atomic>
#include <ctime>

// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::sort. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::cuda::malloc.
//
// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

struct BlazingCachedAllocation{

	char * allocatedSpace;
	std::ptrdiff_t numBytes;
	std::clock_t lastAccessed;
};
// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;

    cached_allocator(size_t maxSize) {
    	this->maxSize =  maxSize;
    	this->curConsumption = 0;
    }

    ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
    	bool allocated = false;
    	BlazingCachedAllocation * allocation;
    	// search the cache for a free block
    	while ( ! allocated){
    		allocMutex.lock();
    		free_blocks_type::iterator free_block = free_blocks.lower_bound(num_bytes);

    		if(free_block != free_blocks.end())
    		{
    			//std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

    			// get the pointer

    			allocation = free_block->second;

    			// erase from the free_blocks map
    			free_blocks.erase(free_block);
    			allocMutex.unlock();
    			allocated = true;
    		}
    		else
    		{
    			allocMutex.unlock();
    			//before we allocate we need to make sure there is room  if there isn't we try this all over
    			consumedAmountMutex.lock();
    			//there should be some kind of compare swap operations that can allow this to be lock free
    			if((curConsumption + num_bytes ) < this->maxSize){
    				curConsumption += num_bytes;
    				consumedAmountMutex.unlock();
    				allocated = true;

    				try
    				{
    					//this wont work here has to be lockekd as well
    					//        	while((curConsumption + num_bytes) >  maxSize){
    					//       	  collectGarbage();
    					//        }

    					allocation = new BlazingCachedAllocation();

    					//std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

    					// allocate memory and convert cuda::pointer to raw pointer
    					allocation->numBytes = num_bytes;
    					cudaMalloc((void **) &allocation->allocatedSpace, num_bytes);

    					allocated = true;
    					//result = thrust::cuda::malloc<char>(num_bytes).get();
    				}
    				catch(std::runtime_error &e)
    				{
    					throw;
    				}

    			}else{
    				consumedAmountMutex.unlock();
    				std::cout<<"Going again!"<<std::endl;
    			}


    			// no allocation of the right size exists
    			// create a new one with cuda::malloc
    			// throw if cuda::malloc can't satisfy the request


    		}

    	}


      // insert the allocated pointer into the allocated_blocks map
      consumedMutex.lock();
     // allocation->lastAccessed = std::clock();
      allocated_blocks.insert(std::make_pair(allocation->allocatedSpace, allocation));
      consumedMutex.unlock();
      return allocation->allocatedSpace;
    }

    void collectGarbage(){
    	//free blocks that have not been used for a while
    }

    void deallocate(char *ptr, size_t n)
    {
      // erase the allocated block from the allocated blocks map
    	consumedMutex.lock();
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      BlazingCachedAllocation * allocation = iter->second;
      allocated_blocks.erase(iter);
      consumedMutex.unlock();
      // insert the block into the free blocks map
      allocMutex.lock();
      free_blocks.insert(std::make_pair(allocation->numBytes, allocation));
      allocMutex.unlock();
    }

  private:
    typedef std::multimap<std::ptrdiff_t, BlazingCachedAllocation*> free_blocks_type;
    typedef std::map<char *, BlazingCachedAllocation*>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;
    std::mutex allocMutex;
    std::mutex consumedMutex;
    std::mutex consumedAmountMutex;
    size_t maxSize;
    std::atomic_size_t curConsumption;
    void free_all()
    {
    	std::cout<<"Consumption size was "<<this->curConsumption/( 1024 * 1024)<<"MB"<<std::endl;
      //std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

      // deallocate all outstanding blocks in both lists
      for(free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        //thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        cudaFree(i->second->allocatedSpace);
        delete i->second;
      }

      for(allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        cudaFree(i->first);
        delete i->second;
 //   	  thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
    }

};

cached_allocator cachedDeviceAllocator(4000000000);

template<typename T>
  struct BlazingDeviceAllocator : thrust::device_malloc_allocator<T>
{
  // shorthand for the name of the base class
  typedef thrust::device_malloc_allocator<T> super_t;

  // get access to some of the base class's typedefs

  // note that because we inherited from device_malloc_allocator,
  // pointer is actually thrust::device_ptr<T>
  typedef typename super_t::pointer   pointer;

  typedef typename super_t::size_type size_type;

  // customize allocate
  pointer allocate(size_type n)
  {

    // defer to the base class to allocate storage for n elements of type T
    // in practice, you'd do something more interesting here
    return pointer((T *) cachedDeviceAllocator.allocate( n * sizeof(T)));
    //return super_t::allocate(n);
  }

  // customize deallocate
  void deallocate(pointer p, size_type n)
  {

	  cachedDeviceAllocator.deallocate((char *) p.get(), n * sizeof(T));


  }
};


#endif /* BLAZINGCACHEALLOCATOR_H_ */

