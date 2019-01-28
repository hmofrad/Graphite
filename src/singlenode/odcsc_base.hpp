/*
 * odcsc_base.hpp: Base class for Optimized Double Compressed Sparse Column (ODCSC) data structure
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef ODCSC_BASE_HPP
#define ODCSC_BASE_HPP

#include <sys/mman.h>
#include <cstring> 

struct CSCEntry
{
  uint32_t global_idx;
  uint32_t idx;
  uint32_t weight;
};

struct Edge
{
  const uint32_t src, dst;

  const char weight;

  Edge() : src(0), dst(0), weight(1) {}

  Edge(const uint32_t src, const uint32_t dst, const char weight)
      : src(src), dst(dst), weight(weight) {}
};

struct ODCSC_BASE {
    public:
        ODCSC_BASE(uint64_t nnz_, uint32_t nnzcols_);
        ~ODCSC_BASE();
        uint64_t nnz;
        uint32_t nnzcols;
        uint64_t size;
        void* ENTRIES;  // ENTRIES
        void* JA; // COL_PTR
        void* JC; // COL_IDX
};

ODCSC_BASE::ODCSC_BASE(uint64_t nnz_, uint32_t nnzcols_) {
    nnz = nnz_;
    nnzcols = nnzcols_;
    
    if((ENTRIES = mmap(nullptr, nnz * sizeof(CSCEntry), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
        fprintf(stderr, "Error mapping memory\n");
        exit(1);
    }
    memset(ENTRIES, 0, nnz * sizeof(CSCEntry));
    
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

    size = (nnz * sizeof(CSCEntry)) + ((nnzcols + 1) * sizeof(uint32_t)) + (nnzcols * sizeof(uint32_t));
}

ODCSC_BASE::~ODCSC_BASE() {
    if(munmap(ENTRIES, nnz * sizeof(CSCEntry)) == -1) {
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