/*
 * numa_vector.hpp: NUMA-aware vector storage implementaion
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef NUMA_VECTOR_HPP
#define NUMA_VECTOR_HPP

template<typename Integer_Type>
struct blk {
    Integer_Type nitems;
    uint64_t nbytes;
    int socket_id;
};

template<typename Integer_Type, typename Vector_Type>
void allocate_numa_vector(Vector_Type*** data, std::vector<struct blk<Integer_Type>>& blks) {
//void allocate_numa_vector(Vector_Type*** data, const std::vector<Integer_Type> nitems, const std::vector<int32_t> socket_ids, std::vector<uint64_t>& bytes) {    
    int32_t vector_length = blks.size();
    uint64_t nbytes = 0;
    uint64_t alignment = 0;
    if(numa_available() != -1) {
        nbytes = vector_length * sizeof(Vector_Type*);
        *data = (Vector_Type**) numa_alloc_onnode(nbytes, Env::socket_id);
        memset(*data, 0, nbytes);
        
        for(int32_t i = 0; i < vector_length; i++) {
            auto& blk = blks[i];
            if(blk.nitems) {
                //nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes = blk.nitems * sizeof(Vector_Type);
                alignment = Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE);
                nbytes += alignment;
                blk.nbytes = nbytes;
                //bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) numa_alloc_onnode(nbytes, blk.socket_id);
                memset((*data)[i], 0, nbytes);  
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);                
            }
            else {
                (*data)[i] = nullptr;
                blk.nbytes = 0;
            }
        }   
    }
    else {        
        nbytes = vector_length * sizeof(Vector_Type*);
        *data = (Vector_Type**) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        memset(*data, 0, nbytes);
        for(int32_t i = 0; i < vector_length; i++) {
            auto& blk = blks[i];
            if(blk.nitems) {
                auto& blk = blks[i];
                //nbytes = nitems[i] * sizeof(Vector_Type);
                nbytes = blk.nitems * sizeof(Vector_Type);
                alignment += (Env::L1_CACHE_LINE_SIZE - (nbytes % Env::L1_CACHE_LINE_SIZE));
                nbytes += alignment;
                blk.nbytes = nbytes;
                //bytes[i] = nbytes;
                (*data)[i] = (Vector_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                memset((*data)[i], 0, nbytes);
                madvise((*data)[i], nbytes, MADV_SEQUENTIAL);   
            }
            else {
                (*data)[i] = nullptr;
                blk.nbytes = 0;
            }
        }
    }
} 

template<typename Integer_Type, typename Vector_Type>
void deallocate_numa_vector(Vector_Type*** data, std::vector<struct blk<Integer_Type>> blks){
//void deallocate_numa_vector(Vector_Type*** data, const std::vector<Integer_Type> nitems, const std::vector<uint64_t> bytes){
    //int32_t vector_length = nitems.size();
    int32_t vector_length = blks.size();
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        for(int32_t i = 0; i < vector_length; i++) {
            auto& blk = blks[i];
            if(blk.nitems) {
                nbytes = blk.nbytes;
                memset((*data)[i], 0, nbytes);    
                numa_free((*data)[i], nbytes);
            }
        }
        nbytes = vector_length * sizeof(Vector_Type*);
        memset(*data, 0, nbytes);    
        numa_free(*data, nbytes);
    }
    else {
        for(int32_t i = 0; i < vector_length; i++) {
            auto& blk = blks[i];
            if(blk.nitems) {
                nbytes = blk.nbytes;
                memset((*data)[i], 0, nbytes);    
                munmap((*data)[i], nbytes);
            }
        }
        nbytes = vector_length * sizeof(Vector_Type*);
        memset(*data, 0, nbytes);
        munmap(*data, nbytes);
    }
}
#endif


