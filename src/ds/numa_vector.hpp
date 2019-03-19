/*
 * numa_vector.hpp: NUMA-aware vector storage implementaion
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef NUMA_VECTOR_HPP
#define NUMA_VECTOR_HPP

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
                
                /*
                nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes += (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                memset((*data)[i], 0, nbytes);
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);
                long off = __sync_fetch_and_add(&((*data)[i]), bytes[i]);
                if(!Env::rank) {
                    printf("%lu %p %p %p\n", nbytes, (void*) (*data)[i], (void*) off, (void*) ((*data)[i] + off));
                }
                
                
                //nbytes = offset + ((align - (offset mod align)) mod align)
                //if(!Env::rank)
                //    printf("0.%lu \n", nbytes);
                //pint *ret = (pint*)((((pint)ptr + sizeof(pint)) & ~(pint)(align - 1)) + align);

                //nbytes += (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                nbytes += Env::L1_CACHE_LINE_SIZE;
                bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                if(!Env::rank)
                    printf("1.%lu %p %p\n", nbytes, (void*) (*data)[i], (Vector_Type*) ((*data)[i] + nbytes));
                memset((*data)[i], 0, nbytes);
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);
                
                //Vector_Type* off = (Vector_Type*) __sync_fetch_and_add(&((*data)[i]), bytes[i]);
                //Vector_Type*  off = (Vector_Type*) __sync_fetch_and_add(&((*data)[i]),size) + bytes[i] > nm->bytes
                if(!Env::rank) {
                    //printf("%lu %p %p %p %lu\n", nbytes, (void*) (*data)[i], (void*) off, (void*) (off + nbytes), sizeof(Vector_Type));
                    //long offset = (long) (*data)[i];
                    //long padding = (Env::L1_CACHE_LINE_SIZE - (offset & (Env::L1_CACHE_LINE_SIZE - 1))) & (Env::L1_CACHE_LINE_SIZE - 1);
                    //long pint;
                    uint32_t* ret = (uint32_t*)((((uint32_t)(*data)[i]) & ~(uint32_t)(Env::L1_CACHE_LINE_SIZE - 1)) + Env::L1_CACHE_LINE_SIZE);

                    //long padding = (Env::L1_CACHE_LINE_SIZE - ((*data)[i] % (void*) Env::L1_CACHE_LINE_SIZE)) % Env::L1_CACHE_LINE_SIZE;
                    printf("%p\n", ret);
                    //for(int i = 0; i < nitems[i]; i++) {
                        
                    //}
                }
                
                
                
                Env::barrier();
                Env::exit(0);
                */
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
    
    
    
  //*arr = (int**)malloc(n*sizeof(int*));
  //for(int i=0; i<n; i++)
    //(*arr)[i] = (int*)malloc(m*sizeof(int));
} 

template<typename Integer_Type, typename Vector_Type>
void deallocate_numa_vector(Vector_Type*** data, const std::vector<Integer_Type> nitems, const std::vector<uint64_t> bytes){
    int32_t vector_length = nitems.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        for(int32_t i = 0; i < vector_length; i++) {
            //nbytes = nitems[i] * sizeof(Vector_Type);
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
                //nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes = bytes[i];
                memset((*data)[i], 0, nbytes);    
                munmap((*data)[i], nbytes);
            }
        }
        nbytes = vector_length * sizeof(Vector_Type*);
        memset(*data, 0, nbytes);
        munmap(*data, nbytes);
    }
    
    
    
    //for (int i = 0; i < n; i++)
      //  free((*arr)[i]);
    //free(*arr); 
}


#endif


