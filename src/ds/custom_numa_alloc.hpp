/*
 * custom_numa_alloc.hpp: NUMA-aware storage implementaion
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CUSTOM_NUMA_ALLOC_HPP
#define CUSTOM_NUMA_ALLOC_HPP


template <typename Vector_Type>
struct Numa_Allocator {
    using value_type = Vector_Type;
    Numa_Allocator() = default;
    std::size_t alignment;
    int socket_id;
    //template <typename U>
    //Numa_Allocator(const Numa_Allocator<U>&) {}
    template <typename U>
    Numa_Allocator(const Numa_Allocator<U>&){}
    
    Numa_Allocator(int socket_id_) : socket_id(socket_id_) {}// printf("socket_id=%d\n", socket_id);
    Vector_Type* allocate(std::size_t n) {
        //std::cout << "Allocating " << n << std::endl;
        //void* d = (void*) mmap(nullptr, 1, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        //return static_cast<T*>(::operator new(n * sizeof(T)));
        //return (T*) mmap(nullptr, (n * sizeof(T)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if(!Env::rank) printf("Allocating %d\n", socket_id);
        uint64_t nbytes = 0;
        void *p;
        if(numa_available() != -1) {
            if(n) {
                uint64_t nbytes = n * sizeof(Vector_Type);
                alignment = (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                nbytes += alignment;
                p = (Vector_Type*) numa_alloc_onnode(nbytes, socket_id);
                memset(p, 0, nbytes);  
                madvise(p, nbytes, MADV_SEQUENTIAL);  
            }
        }
        else {
            
            
            if(n) {
                uint64_t nbytes = n * sizeof(Vector_Type);
                alignment = (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                nbytes += alignment;
                p = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                //if(!Env::rank) printf("2. %d \n", alignment);
                //bytes[i] = nbytes;
                //(*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                memset(p, 0, nbytes);
                madvise(p, nbytes, MADV_SEQUENTIAL);   
            }
            //else {
              //  (*data)[i] = nullptr;
               // bytes[i] = 0;
            //}
            
            
        }
        
        
        //return (Vector_Type*) malloc(n * sizeof(Vector_Type));
        return((Vector_Type*) p);
        
    }
    void deallocate(Vector_Type* p, std::size_t n) {
        if(!Env::rank) printf("Deallocating %d\n", socket_id);
        uint64_t nbytes = 0;
        if(numa_available() != -1) {
            if(n) {
                nbytes = (n * sizeof(Vector_Type)) + alignment;
                memset(p, 0, nbytes);    
                numa_free(p, nbytes);
            }
        }
        else {
            if(n) {
                nbytes = (n * sizeof(Vector_Type)) + alignment;
                memset(p, 0, nbytes);    
                munmap(p, nbytes);
            }
        }
    }
};
template <typename Vector_Type, typename U>
bool operator==(const Numa_Allocator<Vector_Type>&, const Numa_Allocator<U>&) { return true; }
template <typename Vector_Type, typename U>
bool operator!=(const Numa_Allocator<Vector_Type>&, const Numa_Allocator<U>&) { return false; }




template<typename Integer_Type, typename Vector_Type>
struct blk {
    Vector_Type* data;
    Integer_Type nitems;
    uint64_t nbytes;
};
/*
template<typename Integer_Type, typename Vector_Type>
void allocate_numa_vector(Vector_Type*** data, const std::vector<Integer_Type> nitems, const std::vector<int32_t> socket_ids, std::vector<uint64_t>& bytes) {
    int32_t vector_length = nitems.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        nbytes = vector_length * sizeof(Vector_Type*);
        *data = (Vector_Type**) numa_alloc_onnode(nbytes, Env::socket_id);
        memset(*data, 0, nbytes);
        
        for(int32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes += Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE);
                bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) numa_alloc_onnode(nbytes, socket_ids[i]);
                memset((*data)[i], 0, nbytes);  
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);                
            }
            else {
                (*data)[i] = nullptr;
                bytes[i] = 0;
            }
        }   
    }
    else {        
        nbytes = vector_length * sizeof(Vector_Type*);
        *data = (Vector_Type**) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        memset(*data, 0, nbytes);
        for(int32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes += (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                memset((*data)[i], 0, nbytes);
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);   
            }
            else {
                (*data)[i] = nullptr;
                bytes[i] = 0;
            }
        }
    }
} 

template<typename Integer_Type, typename Vector_Type>
void deallocate_numa_vector(Vector_Type*** data, const std::vector<Integer_Type> nitems, const std::vector<uint64_t> bytes){
    int32_t vector_length = nitems.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        for(int32_t i = 0; i < vector_length; i++) {
            nbytes = bytes[i];
            memset((*data)[i], 0, nbytes);    
            numa_free((*data)[i], nbytes);
        }
        nbytes = vector_length * sizeof(Vector_Type*);
        memset(*data, 0, nbytes);    
        numa_free(*data, nbytes);
    }
    else {
        for(int32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = bytes[i];
                memset((*data)[i], 0, nbytes);    
                munmap((*data)[i], nbytes);
            }
        }
        nbytes = vector_length * sizeof(Vector_Type*);
        memset(*data, 0, nbytes);
        munmap(*data, nbytes);
    }
}
*/

#endif


