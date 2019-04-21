/*
 * compressed_column.hpp: Column compressed storage implementaion
 * Triply Compressed Sparse Column (TCSC)
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMPRESSED_STORAGE_HPP
#define COMPRESSED_STORAGE_HPP

#include <sys/mman.h>
#include <cstring> 
 
enum Compression_type
{
  _TCSC_,    // Triply Compressed Sparse Column
  _TCSC_CF_, // Triply Compressed Sparse Column - Computation Filtering
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
                              
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,    
                              const char*         nnzrows_bitvector,
                              const Integer_Type* nnzrows_indices, 
                              const char*         nnzcols_bitvector,
                              const Integer_Type* nnzcols_indices,                
                              const std::vector<Integer_Type>& regular_rows_indices,
                              const std::vector<char>&         regular_rows_bitvector,
                              const std::vector<Integer_Type>& source_rows_indices,
                              const std::vector<char>&         source_rows_bitvector,
                              const std::vector<Integer_Type>& regular_cols_indices,
                              const std::vector<char>&         regular_cols_bitvector,
                              const std::vector<Integer_Type>& sink_cols_indices,
                              const std::vector<char>&         sink_cols_bitvector,
                              const int socket_id){};                              
};

template<typename Weight, typename Integer_Type>
struct TCSC_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        TCSC_BASE(const uint64_t nnz_, const Integer_Type nnzcols_, const Integer_Type nnzrows_, const int socket_id = 0);
        ~TCSC_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,    
                              const char*         nnzrows_bitvector,
                              const Integer_Type* nnzrows_indices, 
                              const char*         nnzcols_bitvector,
                              const Integer_Type* nnzcols_indices);                        
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
        /* JC and IR are allocted per row and column groups in matrix.hpp */
};

template<typename Weight, typename Integer_Type>
TCSC_BASE<Weight, Integer_Type>::TCSC_BASE(const uint64_t nnz_, const Integer_Type nnzcols_, const Integer_Type nnzrows_, const int socket_id) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    nnzrows = nnzrows_;
    if(nnz and nnzcols and nnzrows) {
        if(Env::numa_allocation) {
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
        if(Env::numa_allocation) {    
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

template<typename Weight, typename Integer_Type>
struct TCSC_CF_BASE : public Compressed_column<Weight, Integer_Type> {
    public:
        TCSC_CF_BASE(const uint64_t nnz_,const Integer_Type nnzcols_,const Integer_Type nnzrows_, const int socket_id);
        ~TCSC_CF_BASE();
        virtual void populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                              const Integer_Type tile_height, 
                              const Integer_Type tile_width,                
                              const char*         nnzrows_bitvector,
                              const Integer_Type* nnzrows_indices, 
                              const char*         nnzcols_bitvector,
                              const Integer_Type* nnzcols_indices,               
                              const std::vector<Integer_Type>& regular_rows_indices,
                              const std::vector<char>&         regular_rows_bitvector,
                              const std::vector<Integer_Type>& source_rows_indices,
                              const std::vector<char>&         source_rows_bitvector,
                              const std::vector<Integer_Type>& regular_cols_indices,
                              const std::vector<char>&         regular_cols_bitvector,
                              const std::vector<Integer_Type>& sink_cols_indices,
                              const std::vector<char>&         sink_cols_bitvector,
                              const int socket_id);
                              
        void allocate_local_reg(Integer_Type nnzcols_regulars_local_);
        void allocate_local_src(Integer_Type nnzcols_sources_local_);
        void allocate_local_src_reg(Integer_Type nnzcols_sources_regulars_local_);
        void allocate_local_reg_snk(Integer_Type nnzcols_regulars_sinks_local_);
        void allocate_local_src_snk(Integer_Type nnzcols_sources_sinks_local_);
        uint64_t nnz;
        Integer_Type nnzcols;
        Integer_Type nnzcols_regulars;
        Integer_Type nnzcols_regulars_local;
        Integer_Type nnzcols_sources_local;
        Integer_Type nnzcols_sources_regulars_local;
        Integer_Type nnzcols_regulars_sinks_local;
        Integer_Type nnzcols_sources_sinks_local;
        Integer_Type nnzrows;
        #ifdef HAS_WEIGHT
        Weight* A;  // WEIGHT
        #endif
        Integer_Type* IA; // ROW_IDX
        Integer_Type* JA; // COL_PTR
        Integer_Type* JC; // COL_IDX
        Integer_Type* IR; // ROW_PTR
        Integer_Type* JA_REG_R_NNZ_C;
        Integer_Type* JC_REG_R_NNZ_C;
        Integer_Type  NC_REG_R_REG_C;
        Integer_Type* JA_REG_R_REG_C;
        Integer_Type* JC_REG_R_REG_C;
        Integer_Type  NC_REG_R_SNK_C;
        Integer_Type* JA_REG_R_SNK_C;
        Integer_Type* JC_REG_R_SNK_C;
        Integer_Type  NC_SRC_R_REG_C;
        Integer_Type* JA_SRC_R_REG_C;
        Integer_Type* JC_SRC_R_REG_C;
        Integer_Type  NC_SRC_R_SNK_C;
        Integer_Type* JA_SRC_R_SNK_C;
        Integer_Type* JC_SRC_R_SNK_C;
};

template<typename Weight, typename Integer_Type>
TCSC_CF_BASE<Weight, Integer_Type>::TCSC_CF_BASE(const uint64_t nnz_, const Integer_Type nnzcols_, const Integer_Type nnzrows_, const int socket_id) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    nnzrows = nnzrows_;
    if(nnz and nnzcols and nnzrows) {
        if(Env::numa_allocation) {
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
            
            JC = (Integer_Type*) numa_alloc_onnode(nnzcols * sizeof(Integer_Type), socket_id);
            memset(JC, 0, nnzcols * sizeof(Integer_Type));
            madvise(JC, nnzcols * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            IR = (Integer_Type*) numa_alloc_onnode(nnzrows * sizeof(Integer_Type), socket_id);
            memset(IR, 0, nnzrows * sizeof(Integer_Type));
            madvise(IR, nnzrows * sizeof(Integer_Type), MADV_SEQUENTIAL);
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
            
            if((JC = (Integer_Type*) mmap(nullptr, nnzcols * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JC, 0, nnzcols * sizeof(Integer_Type));
            madvise(JC, nnzcols * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((IR = (Integer_Type*) mmap(nullptr, nnzrows * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(IR, 0, nnzrows * sizeof(Integer_Type));
            madvise(IR, nnzrows * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }
}

template<typename Weight, typename Integer_Type>
TCSC_CF_BASE<Weight, Integer_Type>::~TCSC_CF_BASE() {
    if(nnz and nnzcols and nnzrows) {
        if(Env::numa_allocation) {
            #ifdef HAS_WEIGHT
            numa_free(A, (nnz * sizeof(Weight)));
            #endif
            numa_free(IA, (nnz * sizeof(Integer_Type)));
            numa_free(JA, ((nnzcols + 1) * sizeof(Integer_Type)));
            numa_free(JC, (nnzcols * sizeof(Integer_Type)));
            numa_free(IR, (nnzrows * sizeof(Integer_Type)));
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
            
            if(munmap(JC, nnzcols * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            
            if(munmap(IR, nnzrows * sizeof(Integer_Type)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
            }
        }
    }
    
    if(nnzcols) {
        if(Env::numa_allocation) {
            numa_free(JA_REG_R_NNZ_C, (nnzcols * 2) * sizeof(Integer_Type));   
        }
        else {
            if(munmap(JA_REG_R_NNZ_C, (nnzcols * 2) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }    
   
    if(NC_REG_R_REG_C) {
        if(Env::numa_allocation) {
            numa_free(JA_REG_R_REG_C, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type));   
            numa_free(JC_REG_R_REG_C, NC_REG_R_REG_C * sizeof(Integer_Type));   
        }
        else {
            if(munmap(JA_REG_R_REG_C, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        
            if(munmap(JC_REG_R_REG_C, NC_REG_R_REG_C * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }
    
    if(NC_REG_R_SNK_C) {
        if(Env::numa_allocation) {
            numa_free(JA_REG_R_SNK_C, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type));   
            numa_free(JC_REG_R_SNK_C, NC_REG_R_SNK_C * sizeof(Integer_Type));   
        }
        else {
            if(munmap(JA_REG_R_SNK_C, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            
            if(munmap(JC_REG_R_SNK_C, NC_REG_R_SNK_C * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }

    if(NC_SRC_R_REG_C) {
        if(Env::numa_allocation) {
            numa_free(JA_SRC_R_REG_C, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type));   
            numa_free(JC_SRC_R_REG_C, NC_SRC_R_REG_C * sizeof(Integer_Type));   
        }
        else {
            if(munmap(JA_SRC_R_REG_C, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            
            if(munmap(JC_SRC_R_REG_C, NC_SRC_R_REG_C * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }
    
    if(NC_SRC_R_SNK_C) {
        if(Env::numa_allocation) {
            numa_free(JA_SRC_R_SNK_C, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type));   
            numa_free(JC_SRC_R_SNK_C, NC_SRC_R_SNK_C * sizeof(Integer_Type));   
        }
        else {
            if(munmap(JA_SRC_R_SNK_C, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
            
            if(munmap(JC_SRC_R_SNK_C, NC_SRC_R_SNK_C * sizeof(Integer_Type)) == -1) {
                fprintf(stderr, "Error unmapping memory\n");
                exit(1);
            }
        }
    }
    
}

template<typename Weight, typename Integer_Type>
void TCSC_CF_BASE<Weight, Integer_Type>::populate(const std::vector<struct Triple<Weight, Integer_Type>>* triples,
                                               const Integer_Type tile_height, 
                                               const Integer_Type tile_width,                
                                               const char*         nnzrows_bitvector,
                                               const Integer_Type* nnzrows_indices, 
                                               const char*         nnzcols_bitvector,
                                               const Integer_Type* nnzcols_indices,     
                                               const std::vector<Integer_Type>& regular_rows_indices,
                                               const std::vector<char>&         regular_rows_bitvector,
                                               const std::vector<Integer_Type>& source_rows_indices,
                                               const std::vector<char>&         source_rows_bitvector,
                                               const std::vector<Integer_Type>& regular_cols_indices,
                                               const std::vector<char>&         regular_cols_bitvector,
                                               const std::vector<Integer_Type>& sink_cols_indices,
                                               const std::vector<char>&         sink_cols_bitvector,
                                               const int socket_id) {
    if(not(nnz and nnzcols and nnzrows)) {
        return;
    }        
    
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
    // Column indices
    Integer_Type k = 0;
    for(j = 0; j < tile_height; j++) {
        if(nnzcols_bitvector[j]) {
            JC[k] = j;
            k++;
        }
    }
    assert(nnzcols == k);
    // Local columns
    Integer_Type nnzcols_local = 0;
    std::vector<Integer_Type> JC_LOCAL_VAL;
    std::vector<Integer_Type> JC_LOCAL_IDX;
    for(j = 0; j < nnzcols; j++) {	
        if(JA[j] != JA[j + 1]) {
            JC_LOCAL_VAL.push_back(JC[j]);		
            JC_LOCAL_IDX.push_back(j);	
            nnzcols_local++;            
        }
    }
    // Rows indices
    k = 0;
    for(i = 0; i < tile_height; i++) {
        if(nnzrows_bitvector[i]) {
            IR[k] = i;
            k++;
        }
    }
    assert(nnzrows == k);
    // Moving source rows to the end
    Integer_Type l = 0;
    Integer_Type m = 0;
    Integer_Type n = 0;
    Integer_Type o = 0;
    std::vector<Integer_Type> r;
    for(j = 0; j < nnzcols; j++) {
        for(i = JA[j]; i < JA[j + 1]; i++) {
            if(source_rows_bitvector[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            if(m > n) {
                for(Integer_Type p = 0; p < n; p++) {
                    for(Integer_Type q = JA[j+1] - 1; q >= JA[j]; q--) {
                        if(source_rows_bitvector[IR[IA[q]]] != 1) {
                            #ifdef HAS_WEIGHT
                            std::swap(A[r[p]], A[q]);
                            #endif
                            std::swap(IA[r[p]], IA[q]);
                            break;
                        }
                        else {
                            if(r[p] == q)
                                break;
                        }
                    }
                }
            }
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
    }
    
    if(nnzcols) {
        if(Env::numa_allocation) {
            JA_REG_R_NNZ_C = (Integer_Type*) numa_alloc_onnode((nnzcols * 2) * sizeof(Integer_Type), socket_id);
            memset(JA_REG_R_NNZ_C, 0, (nnzcols * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_NNZ_C, (nnzcols * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
        else {        
            if((JA_REG_R_NNZ_C = (Integer_Type*) mmap(nullptr, (nnzcols * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA_REG_R_NNZ_C, 0, (nnzcols * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_NNZ_C, (nnzcols * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }
    
    // Regular rows to nnz columns
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit();  
    for(j = 0; j < nnzcols; j++) {
        for(i = JA[j]; i < JA[j + 1]; i++) {
            if(source_rows_bitvector[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }
        }
        if(m > 0) {              
            n = r.size();
            JA_REG_R_NNZ_C[l] = JA[j];
            JA_REG_R_NNZ_C[l + 1] = JA[j + 1] - n;
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_R_NNZ_C[l] = JA[j];
            JA_REG_R_NNZ_C[l + 1] = JA[j + 1];
            l += 2;  
        }
    }

    // Regular rows to regular columns
    Integer_Type j1 = 0;
    Integer_Type j2 = 0;
    k = 0;
    l = 0;
    m = 0;
    if(regular_cols_indices.size()) {
        while((j1 < nnzcols_local) and (j2 < regular_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == regular_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1)
                        k++;
                }
                if(k) {
                    m = JA[j + 1] - JA[j];
                    if(m == k)
                        l--;
                    k = 0;
                    m = 0;
                }
                l++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < regular_cols_indices[j2])
                j1++;
            else
                j2++;
        }
    }

    NC_REG_R_REG_C = l;
    if(NC_REG_R_REG_C) {
        if(Env::numa_allocation) {
            JA_REG_R_REG_C = (Integer_Type*) numa_alloc_onnode((NC_REG_R_REG_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JA_REG_R_REG_C, 0, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_REG_C, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            JC_REG_R_REG_C = (Integer_Type*) numa_alloc_onnode((NC_REG_R_REG_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JC_REG_R_REG_C, 0, NC_REG_R_REG_C * sizeof(Integer_Type));
            madvise(JC_REG_R_REG_C, NC_REG_R_REG_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
        else {          
            if((JA_REG_R_REG_C = (Integer_Type*) mmap(nullptr, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA_REG_R_REG_C, 0, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_REG_C, (NC_REG_R_REG_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((JC_REG_R_REG_C = (Integer_Type*) mmap(nullptr, NC_REG_R_REG_C * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JC_REG_R_REG_C, 0, NC_REG_R_REG_C * sizeof(Integer_Type));
            madvise(JC_REG_R_REG_C, NC_REG_R_REG_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }
    
    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit();     
    if(NC_REG_R_REG_C) {
        
        while((j1 < nnzcols_local) and (j2 < regular_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == regular_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1) {
                        k++;
                        m = (JA[j+1] - JA[j]);
                        r.push_back(i);
                    }   
                }
                if(m > 0) {
                    if(m != k) {                                            
                        n = r.size();
                        JA_REG_R_REG_C[l] = JA[j];
                        JA_REG_R_REG_C[l + 1] = JA[j + 1] - n;
                        l += 2; 
                        JC_REG_R_REG_C[o] = j;
                        o++;
                    }
                    k = 0;
                    m = 0;
                    n = 0;
                    r.clear();
                    r.shrink_to_fit();
                }
                else {
                    JA_REG_R_REG_C[l] = JA[j];
                    JA_REG_R_REG_C[l + 1] = JA[j + 1];
                    l += 2;  
                    JC_REG_R_REG_C[o] = j;
                    o++;
                }
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < regular_cols_indices[j2])
                j1++;
            else
                j2++;
            
        }        
    }

    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;
    m = 0;
    if(sink_cols_indices.size()) {
        while((j1 < nnzcols_local) and (j2 < sink_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == sink_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1)
                        k++;
                }
                if(k) {
                    m = JA[j + 1] - JA[j];
                    if(m == k)
                        l--;
                    k = 0;
                    m = 0;
                }
                l++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < sink_cols_indices[j2])
                j1++;
            else
                j2++;
        }
    }

    // Regular rows to sink columns
    NC_REG_R_SNK_C = l;
    if(NC_REG_R_SNK_C) {
        if(Env::numa_allocation) {
            JA_REG_R_SNK_C = (Integer_Type*) numa_alloc_onnode((NC_REG_R_SNK_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JA_REG_R_SNK_C, 0, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_SNK_C, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            JC_REG_R_SNK_C = (Integer_Type*) numa_alloc_onnode((NC_REG_R_SNK_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JC_REG_R_SNK_C, 0, NC_REG_R_SNK_C * sizeof(Integer_Type));
            madvise(JC_REG_R_SNK_C, NC_REG_R_SNK_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
        else {     
            if((JA_REG_R_SNK_C = (Integer_Type*) mmap(nullptr, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA_REG_R_SNK_C, 0, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type));
            madvise(JA_REG_R_SNK_C, (NC_REG_R_SNK_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((JC_REG_R_SNK_C = (Integer_Type*) mmap(nullptr, NC_REG_R_SNK_C * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JC_REG_R_SNK_C, 0, NC_REG_R_SNK_C * sizeof(Integer_Type));
            madvise(JC_REG_R_SNK_C, NC_REG_R_SNK_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }
    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    o = 0;
    r.clear();
    r.shrink_to_fit(); 
    if(NC_REG_R_SNK_C) {
        while((j1 < nnzcols_local) and (j2 < sink_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == sink_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1) {
                        k++;
                        m = (JA[j+1] - JA[j]);
                        r.push_back(i);
                    }   
                }
                if(m > 0) {
                    if(m != k) {                                            
                        n = r.size();
                        JA_REG_R_SNK_C[l] = JA[j];
                        JA_REG_R_SNK_C[l + 1] = JA[j + 1] - n;
                        l += 2; 
                        JC_REG_R_SNK_C[o] = j;
                        o++;
                    }
                    k = 0;
                    m = 0;
                    n = 0;
                    r.clear();
                    r.shrink_to_fit();
                }
                else {
                    JA_REG_R_SNK_C[l] = JA[j];
                    JA_REG_R_SNK_C[l + 1] = JA[j + 1];
                    l += 2;  
                    JC_REG_R_SNK_C[o] = j;
                    o++;
                }
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < sink_cols_indices[j2])
                j1++;
            else
                j2++;
            
        }     
    }
    
    // Source rows to regular columns
    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;
    if(regular_cols_indices.size()) {
        while((j1 < nnzcols_local) and (j2 < regular_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == regular_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1) {
                        k++;
                        break;
                    }
                }
                //k++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < regular_cols_indices[j2])
                j1++;
            else
                j2++;
        }
    }
    
    NC_SRC_R_REG_C = k;
    if(NC_SRC_R_REG_C) {
        if(Env::numa_allocation) {
            JA_SRC_R_REG_C = (Integer_Type*) numa_alloc_onnode((NC_SRC_R_REG_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JA_SRC_R_REG_C, 0, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type));
            madvise(JA_SRC_R_REG_C, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            JC_SRC_R_REG_C = (Integer_Type*) numa_alloc_onnode((NC_SRC_R_REG_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JC_SRC_R_REG_C, 0, NC_SRC_R_REG_C * sizeof(Integer_Type));
            madvise(JC_SRC_R_REG_C, NC_SRC_R_REG_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
        else {  
            if((JA_SRC_R_REG_C = (Integer_Type*) mmap(nullptr, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA_SRC_R_REG_C, 0, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type));
            madvise(JA_SRC_R_REG_C, (NC_SRC_R_REG_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((JC_SRC_R_REG_C = (Integer_Type*) mmap(nullptr, NC_SRC_R_REG_C * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JC_SRC_R_REG_C, 0, NC_SRC_R_REG_C * sizeof(Integer_Type));
            madvise(JC_SRC_R_REG_C, NC_SRC_R_REG_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }

    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    o = 0;
    r.clear();
    r.shrink_to_fit(); 
    if(NC_SRC_R_REG_C) {
        while((j1 < nnzcols_local) and (j2 < regular_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == regular_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1) {
                        m = (JA[j+1] - JA[j]);
                        r.push_back(i);
                    }   
                }
                if(m > 0) {
                    n = r.size();
                    JA_SRC_R_REG_C[l] = JA[j + 1] - n;
                    JA_SRC_R_REG_C[l + 1] = JA[j + 1];            
                    l += 2; 
                    m = 0;
                    n = 0;
                    r.clear();
                    r.shrink_to_fit();
                    JC_SRC_R_REG_C[o] = j;
                    o++;
                }
                k++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < regular_cols_indices[j2])
                j1++;
            else
                j2++;
            
        }
    }  

    // Source rows to sink columns
    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;
    m = 0;
    if(sink_cols_indices.size()) {
        while((j1 < nnzcols_local) and (j2 < sink_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == sink_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1)
                        k++;
                }
                l++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < sink_cols_indices[j2])
                j1++;
            else
                j2++;
        }
    }
      
    NC_SRC_R_SNK_C = k;
    if(NC_SRC_R_SNK_C) {
        if(Env::numa_allocation) {
            JA_SRC_R_SNK_C = (Integer_Type*) numa_alloc_onnode((NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JA_SRC_R_SNK_C, 0, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type));
            madvise(JA_SRC_R_SNK_C, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            JC_SRC_R_SNK_C = (Integer_Type*) numa_alloc_onnode((NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type), socket_id);
            memset(JC_SRC_R_SNK_C, 0, NC_SRC_R_SNK_C * sizeof(Integer_Type));
            madvise(JC_SRC_R_SNK_C, NC_SRC_R_SNK_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
        else {  
            if((JA_SRC_R_SNK_C = (Integer_Type*) mmap(nullptr, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JA_SRC_R_SNK_C, 0, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type));
            madvise(JA_SRC_R_SNK_C, (NC_SRC_R_SNK_C * 2) * sizeof(Integer_Type), MADV_SEQUENTIAL);
            
            if((JC_SRC_R_SNK_C = (Integer_Type*) mmap(nullptr, NC_SRC_R_SNK_C * sizeof(Integer_Type), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                fprintf(stderr, "Error mapping memory\n");
                exit(1);
            }
            memset(JC_SRC_R_SNK_C, 0, NC_SRC_R_SNK_C * sizeof(Integer_Type));
            madvise(JC_SRC_R_SNK_C, NC_SRC_R_SNK_C * sizeof(Integer_Type), MADV_SEQUENTIAL);
        }
    }
    
    
    j1 = 0;
    j2 = 0;
    k = 0;
    l = 0;   
    m = 0;
    n = 0;
    o = 0;
    r.clear();
    r.shrink_to_fit(); 
    if(NC_SRC_R_REG_C) {
        while((j1 < nnzcols_local) and (j2 < sink_cols_indices.size())) {
            if(JC_LOCAL_VAL[j1] == sink_cols_indices[j2]) {
                j = JC_LOCAL_IDX[j1];
                for(i = JA[j]; i < JA[j + 1]; i++) {
                    if(source_rows_bitvector[IR[IA[i]]] == 1) {
                        m = (JA[j+1] - JA[j]);
                        r.push_back(i);
                    }   
                }
                if(m > 0) {
                    n = r.size();
                    JA_SRC_R_SNK_C[l] = JA[j] + n;
                    JA_SRC_R_SNK_C[l + 1] = JA[j + 1];            
                    l += 2; 
                    m = 0;
                    n = 0;
                    r.clear();
                    r.shrink_to_fit();
                    JC_SRC_R_SNK_C[o] = j;
                    o++;
                }
                k++;
                j1++;
                j2++;
            }
            else if (JC_LOCAL_VAL[j1] < sink_cols_indices[j2])
                j1++;
            else
                j2++;
            
        }
    }

    JC_LOCAL_VAL.clear();
    JC_LOCAL_VAL.shrink_to_fit();
    JC_LOCAL_IDX.clear();
    JC_LOCAL_IDX.shrink_to_fit();
}
#endif