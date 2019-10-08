/*
 * tiling.hpp: Tiling implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILING_HPP
#define TILING_HPP
 
#include <cassert> 
#include <cmath>
 
enum Tiling_type {
    //_2DGP_,  // GraphPad
    _2D_,  // 2DT-Staggered + NUMA
    _NUMA_ // 2DT-Staggered + NUMA
};

class Tiling {    
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Graph;
    template<typename Weight___, typename Integer_Type___, typename Fractional_Type___, typename Vertex_State, typename Vertex_Methods_Impl>
    friend class Vertex_Program;
    
    public:    
        Tiling(uint32_t nranks_, uint32_t rank_nthreads_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, Tiling_type tiling_type_);
        ~Tiling();
    
    private:
        Tiling_type tiling_type;
        uint32_t ntiles, nrowgrps, ncolgrps;
        uint32_t nranks;
        uint32_t rank_ntiles, rank_nrowgrps, rank_ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nthreads;
        uint32_t nthreads;
        uint32_t thread_ntiles, thread_nrowgrps, thread_ncolgrps;
        uint32_t rowgrp_nthreads, colgrp_nthreads;
        void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
};

Tiling::Tiling(uint32_t nranks_, uint32_t rank_nthreads_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, Tiling_type tiling_type_) {
    tiling_type = tiling_type_;
    nranks = nranks_;
    ntiles = ntiles_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    rank_ntiles = ntiles / nranks;
    assert(rank_ntiles * nranks == ntiles);
    
    if(not Env::get_init_status()) {
        Env::shuffle_ranks();
        if (tiling_type == Tiling_type::_NUMA_) {
            bool ret = Env::affinity(); // NUMA computation and communication
        }
    }
    integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
    assert(rowgrp_nranks * colgrp_nranks == nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    
    integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
    assert(rowgrp_nranks * colgrp_nranks == nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    
    //printf("%d %d %d\n", Env::nranks, Env::nthreads, Env::nsegments);
    rank_nthreads = rank_nthreads_;
    nthreads = nranks * rank_nthreads;   
    thread_ntiles = ntiles / nthreads;
    assert(thread_ntiles * nthreads == ntiles); 

    integer_factorize(nthreads, rowgrp_nthreads, colgrp_nthreads);
    assert(rowgrp_nthreads * colgrp_nthreads == nthreads);
    thread_nrowgrps = nrowgrps / colgrp_nthreads;
    thread_ncolgrps = ncolgrps / rowgrp_nthreads;        
    assert(thread_nrowgrps * thread_ncolgrps == thread_ntiles);
    
    /*
    //if ((tiling_type == Tiling_type::_2D_) or (tiling_type == Tiling_type::_2DGP_)) {
    if (tiling_type == Tiling_type::_2D_) {
        if(not Env::get_init_status())
            Env::shuffle_ranks();

    }
    else if (tiling_type == Tiling_type::_NUMA_) {
        //rowgrp_nranks = Env::socket_nranks;
        //colgrp_nranks = Env::nmachines * Env::nsockets;
        //rowgrp_nranks = Env::machine_nranks;
        //colgrp_nranks = Env::nmachines;
        
        if(not Env::get_init_status()) {
            Env::shuffle_ranks();
            bool ret = Env::affinity(); // Affinity 
        }
        integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
        assert(rowgrp_nranks * colgrp_nranks == nranks);
        rank_nrowgrps = nrowgrps / colgrp_nranks;
        rank_ncolgrps = ncolgrps / rowgrp_nranks;        
        assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);
    }
    else {
        fprintf(stderr, "ERROR(rank=%d): Invalid tiling type\n", Env::rank);
        Env::exit(1);
    }
    */
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