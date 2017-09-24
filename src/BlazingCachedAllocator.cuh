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
#include <cassert>


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


// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;

    cached_allocator() {}

    ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
      char *result = 0;

      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.lower_bound(num_bytes);

      if(free_block != free_blocks.end())
      {
        //std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
        {
          //std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

          // allocate memory and convert cuda::pointer to raw pointer
        	cudaMalloc((void **) &result, num_bytes);
          //result = thrust::cuda::malloc<char>(num_bytes).get();
        }
        catch(std::runtime_error &e)
        {
          throw;
        }
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_bytes));

      return result;
    }

    void deallocate(char *ptr, size_t n)
    {
      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

  private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all()
    {
      //std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

      // deallocate all outstanding blocks in both lists
      for(free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        //thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        cudaFree(i->second);
      }

      for(allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        cudaFree(i->first);
 //   	  thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
    }

};

cached_allocator cachedDeviceAllocator;

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

