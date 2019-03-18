/*
 * compressed_column.hpp: Column compressed storage implementaion
 * Triply Compressed Sparse Column (TCSC)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP

#include <sys/mman.h>
#include <cstring> 
 
enum Compression_type
{
  _TCSC_, // Triply Compressed Sparse Column
};

template<typename Weight, typename Integer_Type>
struct Compressed_column {
    public:
        virtual ~Compressed_column() {}
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,
                              const char*         nnzrows_bitvector,
                              const Integer_Type* nnzrows_indices, 
                              const char*         nnzcols_bitvector,
                              const Integer_Type* nnzcols_indices) {};
                              //const std::vector<char>&         nnzrows_bitvector,
                              //const std::vector<Integer_Type>& nnzrows_indices, 
                              //const std::vector<char>&         nnzcols_bitvector,
                              //const std::vector<Integer_Type>& nnzcols_indices) {};                           
};

template<typename Weight, typename Integer_Type>
struct TCSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        TCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_, Integer_Type nnzrows_, int socket_id = 0);
        ~TCSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,    
                              const char*         nnzrows_bitvector,
                              const Integer_Type* nnzrows_indices, 
                              const char*         nnzcols_bitvector,
                              const Integer_Type* nnzcols_indices);                        
                              //const std::vector<char>&         nnzrows_bitvector,
                              //const std::vector<Integer_Type>& nnzrows_indices, 
                              //const std::vector<char>&         nnzcols_bitvector,
                              //const std::vector<Integer_Type>& nnzcols_indices);         
        uint64_t nnz;
        Integer_Type nnzcols;
        Integer_Type nnzrows;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
        Integer_Type* JC; // COL_IDX
        Integer_Type* IR; // ROW_PTR
        /* JC and IR are allocted per row and column groups in matrx.hpp*/
};

template<typename Weight, typename Integer_Type>
TCSC_BASE<Weight, Integer_Type>::TCSC_BASE(uint64_t nnz_, Integer_Type nnzcols_, Integer_Type nnzrows_, int socket_id) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    nnzrows = nnzrows_;
    if(nnz and nnzcols and nnzrows) {
        if(numa_available() != -1) {
        //if(0) {            
            #ifdef HAS_WEIGHT
            A = (Weight*) numa_alloc_onnode(nnz * sizeof(Weight), socket_id);
            memset(A, 0, nnz * sizeof(Weight));
            madvise(A, nnz * sizeof(Weight), MADV_SEQUENTIAL);
            #endif
            IA = (Integer_Type*) numa_alloc_onnode(nnz * sizeof(Integer_Type), socket_id);
            memset(IA, 0, nnz * sizeof(Integer_Type));
            madvise(IA, nnz * sizeof(Integer_Type), MADV_SEQUENTIAL);
            JA = (Integer_Type*) numa_alloc_onnode((nnzcols + 1) * sizeof(Integer_Type), socket_id);
            memset(JA, 0, (nnzcols + 1) * sizeof(Integer_Type));
            madvise(JA, (nnzcols + 1) * sizeof(Integer_Type), MADV_SEQUENTIAL);
        } 
        else {
            
            #ifdef HAS_WEIGHT
            if((A = (Weight*) mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(A, 0, nnz * sizeof(Weight));
            madvise(A, nnz * sizeof(Weight), MADV_SEQUENTIAL);
            #endif
            
            if((IA = (Integer_Type*) mmap(nullptr, nnz * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(IA, 0, nnz * sizeof(Integer_Type));
            madvise(IA, nnz * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((JA = (Integer_Type*) mmap(nullptr, (nnzcols + 1) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA, 0, (nnzcols + 1) * sizeof(Integer_Type));
            madvise(JA, (nnzcols + 1) * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }            
    }
}

template<typename Weight, typename Integer_Type>
TCSC_BASE<Weight, Integer_Type>::~TCSC_BASE() {
    if(nnz and nnzcols and nnzrows) {
        
        if(numa_available() != -1) {
        //if(0) {            
            #ifdef HAS_WEIGHT
            numa_free(A, (nnz * sizeof(Weight)));
            #endif
            numa_free(IA, (nnz * sizeof(Integer_Type)));
            numa_free(JA, ((nnzcols + 1) * sizeof(Integer_Type)));
        }
        else {
            #ifdef HAS_WEIGHT
            if(munmap(A, nnz * sizeof(Weight)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            #endif
            if(munmap(IA, nnz * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            
            if(munmap(JA, (nnzcols + 1) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }        
}

template<typename Weight, typename Integer_Type>
void TCSC_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                                               const Integer_Type tile_height, 
                                               const Integer_Type tile_width,                
                                               const char*         nnzrows_bitvector,
                                               const Integer_Type* nnzrows_indices, 
                                               const char*         nnzcols_bitvector,
                                               const Integer_Type* nnzcols_indices) {
                                              // const std::vector<char>&         nnzrows_bitvector,
                                               //const std::vector<Integer_Type>& nnzrows_indices, 
                                               //const std::vector<char>&         nnzcols_bitvector,
                                               //const std::vector<Integer_Type>& nnzcols_indices) {
    if(nnz and nnzcols and nnzrows) {
        struct Triple<Weight, Integer_Type> pair;
        Integer_Type i = 0; // Row Index
        Integer_Type j = 1; // Col index
        JA[0] = 0;
        for (auto& triple : *triples) {
            pair  = {(triple.row % tile_height), (triple.col % tile_width)};
            while((j - 1) != nnzcols_indices[pair.col]) {
                j++;
                JA[j] = JA[j - 1];
            }            
            #ifdef HAS_WEIGHT
            A[i] = triple.weight;
            #endif
            JA[j]++;
            IA[i] = nnzrows_indices[pair.row];
            i++;
        }
        while((j + 1) < (nnzcols + 1)) {
            j++;
            JA[j] = JA[j - 1];
        }
    }
}
#endif