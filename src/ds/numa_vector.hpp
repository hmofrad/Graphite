/*
 * numa_vector.hpp: NUMA-aware vector storage implementaion
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef NUMA_VECTOR_HPP
#define NUMA_VECTOR_HPP

template<typename Integer_Type, typename Vector_Type>
void allocate_numa_vector(Vector_Type*** data, std::vector<Integer_Type> nitems, std::vector<int32_t> socket_ids) {
    int32_t vector_length = nitems.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        nbytes = vector_length * sizeof(Vector_Type*);
        *data = (Vector_Type**) numa_alloc_onnode(nbytes, Env::socket_id);
        memset(*data, 0, nbytes);
        
        for(int32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Vector_Type);
                (*data)[i] = (Vector_Type*) numa_alloc_onnode(nbytes, socket_ids[i]);
                memset((*data)[i], 0, nbytes);    
            }
            else {
                (*data)[i] = nullptr;
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
                (*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                memset((*data)[i], 0, nbytes);
            }
            else {
                (*data)[i] = nullptr;
            }
        }
    }
    
    
    
  //*arr = (int**)malloc(n*sizeof(int*));
  //for(int i=0; i<n; i++)
    //(*arr)[i] = (int*)malloc(m*sizeof(int));
} 

template<typename Integer_Type, typename Vector_Type>
void deallocate_numa_vector(Vector_Type*** data, std::vector<Integer_Type> nitems){
    int32_t vector_length = nitems.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        for(int32_t i = 0; i < vector_length; i++) {
            nbytes = nitems[i] * sizeof(Vector_Type);
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
                nbytes = nitems[i] * sizeof(Vector_Type);
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


