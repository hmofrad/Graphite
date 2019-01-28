/*
 * tiling.hpp: Tiling implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILING_HPP
#define TILING_HPP
 
#include <cassert> 
#include <cmath>
 
enum Tiling_type {
  _2D_,
  _2DT_
};

class Tiling {    
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Graph;
    template<typename Weight___, typename Integer_Type___, typename Fractional_Type___, typename Vertex_State>
    friend class Vertex_Program;
    
    public:    
        Tiling(uint32_t nranks_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, Tiling_type tiling_type_);
        ~Tiling();
    
    private:
        uint32_t nranks;
        uint32_t ntiles, nrowgrps, ncolgrps;
        Tiling_type tiling_type;
        uint32_t rank_ntiles, rank_nrowgrps, rank_ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
};

Tiling::Tiling(uint32_t nranks_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, Tiling_type tiling_type_) {
    nranks = nranks_;
    ntiles = ntiles_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    rank_ntiles = ntiles_ / Env::nranks;
    tiling_type = tiling_type_;
    assert(rank_ntiles * nranks == ntiles);
    if (tiling_type == Tiling_type::_2D_) {
        integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
        assert(rowgrp_nranks * colgrp_nranks == nranks);
        rank_nrowgrps = nrowgrps / colgrp_nranks;
        rank_ncolgrps = ncolgrps / rowgrp_nranks;        
        assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    }
    else if (tiling_type == Tiling_type::_2DT_) {
        integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
        assert(rowgrp_nranks * colgrp_nranks == nranks);
        rank_nrowgrps = nrowgrps / colgrp_nranks;
        rank_ncolgrps = ncolgrps / rowgrp_nranks;        
        assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    }
};

Tiling::~Tiling() {};

void Tiling::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b) {
  /* This approach is adapted from that of GraphPad. */
  a = b = sqrt(n);
  while (a * b != n) {
    b++;
    a = n / b;
  }
  assert(a * b == n);
}
#endif