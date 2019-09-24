/*
 * base_allocator.hpp: Base allocator with support for 
 * multitude of features including NUMU memory allocation, cache alignment, and memory prefetching
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BASE_ALLOCATOR_HPP
#define BASE_ALLOCATOR_HPP

template<typename Integer_Type>
struct blk {
    Integer_Type nitems;
    uint64_t nbytes;
    int socket_id;
};

template<typename Integer_Type, typename Vector_Type>
void allocate(Vector_Type** ptr, struct blk<Integer_Type>& blk, const bool numa_allocation, const bool cache_alignment, const long cache_line_size, const bool memory_prefetching) {
    *ptr = nullptr;
    blk.nbytes = 0;
    bool status = true;
    uint64_t nbytes = 0;
    uint64_t alignment = 0;
    if(blk.nitems) {
        nbytes = blk.nitems * sizeof(Vector_Type);
        if(cache_alignment) {
            alignment += (cache_line_size - (nbytes % cache_line_size));
            nbytes += alignment;
        }
        blk.nbytes = nbytes;
        if(numa_allocation) {
            if((*ptr = (Vector_Type*) numa_alloc_onnode(nbytes, blk.socket_id)) == nullptr) {
                status = false;
            }
        }
        else {
            if((*ptr = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                status = false;
            }
        }
        if(status) {
            memset((void*) *ptr, 0, nbytes);
            if(memory_prefetching) {
                madvise(*ptr, nbytes, MADV_SEQUENTIAL);   
            }
        }
        else {
            fprintf(stderr, "Error allocating memory\n");
            exit(1);
        }
    }
}

template<typename Integer_Type, typename Vector_Type>
void deallocate(Vector_Type** ptr, struct blk<Integer_Type> blk, bool numa_allocation){
    bool status = true;
    uint64_t nbytes = 0;
    if(blk.nitems) {
        nbytes = blk.nbytes;
        memset((void*) *ptr, 0, nbytes);    
        if(numa_allocation) {
            numa_free(*ptr, nbytes);
            if(*ptr == nullptr)
                status = false;
        }
        else {
            if((munmap(*ptr, nbytes)) == -1)
                status = false;
        }
        *ptr = nullptr;
        if(not status) {
            fprintf(stderr, "Error deallocating memory\n");
            exit(1);
        }
    }
}

#endif
