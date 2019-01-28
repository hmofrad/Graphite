/*
 * dcsc_base.hpp: Base class for Double Compressed Sparse Column (DCSC) data structure
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DCSC_BASE_HPP
#define DCSC_BASE_HPP

#include <sys/mman.h>
#include <cstring> 

struct DCSC_BASE {
    public:
        DCSC_BASE(uint64_t nnz_, uint32_t nnzcols_);
        ~DCSC_BASE();
        uint64_t nnz;
        uint32_t nnzcols;
        uint64_t size;
        void* A;  // WEIGHT
        void* IA; // ROW_IDX
        void* JA; // COL_PTR
        void* JC; // COL_IDX
};

DCSC_BASE::DCSC_BASE(uint64_t nnz_, uint32_t nnzcols_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    
    if((A = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(A, 0, nnz * sizeof(uint32_t));
    
    if((IA = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IA, 0, nnz * sizeof(uint32_t));
    
    if((JA = mmap(nullptr, (nnzcols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA, 0, (nnzcols + 1) * sizeof(uint32_t));
    
    if((JC = mmap(nullptr, nnzcols * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC, 0, nnzcols * sizeof(uint32_t));
    
    size = (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint32_t)) + ((nnzcols + 1) * sizeof(uint32_t)) + (nnzcols * sizeof(uint32_t));
}

DCSC_BASE::~DCSC_BASE() {
    if(munmap(A, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(IA, nnz * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA, (nnzcols + 1) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JC, nnzcols * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}
#endif