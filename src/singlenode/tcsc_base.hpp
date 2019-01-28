/*
 * tcsc_base.hpp: Base class for Triple Compressed Sparse Column (TCSC) data structure
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TCSC_BASE_HPP
#define TCSC_BASE_HPP

#include <sys/mman.h>
#include <cstring> 

struct TCSC_BASE {
    public:
        TCSC_BASE(uint64_t nnz_, uint32_t nnzcols_, uint32_t nnzrows_, uint32_t nnzcols_regulars_);
        ~TCSC_BASE();
        uint64_t nnz;
        uint32_t nnzcols;
        uint32_t nnzcols_regulars;
        uint32_t nnzrows;
        uint64_t size;
        void* A;     // WEIGHT
        void* IA;    // ROW_IDX
        void* JA;    // COL_PTR
        void* JC;    // COL_IDX
        void* IR;    // ROW_PTR
        void* JA_REG_C;  // COL_PTR_REG_COL
        void* JC_REG_C;  // COL_IDX_REG_COL
        void* JA_REG_R;  // COL_PTR_REG_ROW
        void* JA_REG_RC; // COL_PTR_REG_COL_REG_ROW
};

TCSC_BASE::TCSC_BASE(uint64_t nnz_, uint32_t nnzcols_, uint32_t nnzrows_, uint32_t nnzcols_regulars_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    nnzrows = nnzrows_;
    nnzcols_regulars = nnzcols_regulars_;
    
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
    
    if((IR = mmap(nullptr, nnzrows * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(IR, 0, nnzrows * sizeof(uint32_t));
    
    if((JA_REG_C = mmap(nullptr, (nnzcols_regulars * 2) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_C, 0, (nnzcols_regulars * 2) * sizeof(uint32_t));
    
    if((JC_REG_C = mmap(nullptr, nnzcols_regulars * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JC_REG_C, 0, nnzcols_regulars * sizeof(uint32_t));

    if((JA_REG_R = mmap(nullptr, (nnzcols * 2) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_R, 0, (nnzcols * 2) * sizeof(uint32_t));
    
    if((JA_REG_RC = mmap(nullptr, (nnzcols_regulars * 2) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(JA_REG_RC, 0, (nnzcols_regulars * 2) * sizeof(uint32_t));
    
    size = (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint32_t)) + ((nnzcols + 1) * sizeof(uint32_t)) + (nnzcols * sizeof(uint32_t)) + (nnzrows * sizeof(uint32_t))
                                    + ((nnzcols_regulars * 2) * sizeof(uint32_t)) + (nnzcols_regulars * sizeof(uint32_t)) 
                                    + (nnzcols * 2) * sizeof(uint32_t) + ((nnzcols_regulars * 2) * sizeof(uint32_t));
                                    
}

TCSC_BASE::~TCSC_BASE() {
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
    
    if(munmap(IR, nnzrows * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA_REG_C, (nnzcols_regulars * 2) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JC_REG_C, nnzcols_regulars * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
     if(munmap(JA_REG_R, (nnzcols * 2) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
    
    if(munmap(JA_REG_RC, (nnzcols_regulars * 2) * sizeof(uint32_t)) == -1) {
        fprintf(stderr, "Error unmapping memory\n");
        exit(1);
    }
}
#endif
