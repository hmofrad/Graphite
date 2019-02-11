/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <omp.h>
 
#include <cmath>
#include <algorithm>
#include <vector>
#include "mpi/types.hpp" 
#include "mat/tiling.hpp" 
#include "ds/indexed_sort.hpp"


enum Filtering_type
{
  _ROWS_,
  _COLS_,
  _NONE_,
  _SOME_
}; 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D { 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight, Integer_Type>>* triples;
    struct Compressed_column<Weight, Integer_Type>* compressor = nullptr;
    uint32_t rg, cg; // Row group, Column group
    uint32_t ith, jth, nth; // ith row, jth column, nth local row order tile,
    uint32_t mth, kth; // mth local column order tile, and kth global tile
    int32_t rank;
    int32_t leader_rank_rg, leader_rank_cg;
    int32_t rank_rg, rank_cg;
    int32_t leader_rank_rg_rg, leader_rank_cg_cg;
    uint64_t nedges;
    void allocate_triples();
    void free_triples();
    int32_t npartitions;
    std::vector<std::vector<struct Triple<Weight, Integer_Type>>*> triples_t;
    std::vector<struct Compressed_column<Weight, Integer_Type>*> compressor_t;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Tile2D<Weight, Integer_Type, Fractional_Type>::allocate_triples() {
    if (!triples)
        triples = new std::vector<struct Triple<Weight, Integer_Type>>;
}
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Tile2D<Weight, Integer_Type, Fractional_Type>::free_triples() {
    triples->clear();
    triples->shrink_to_fit();
    delete triples;
    triples = nullptr;
}
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Matrix {
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__, typename Vertex_state>
    friend class Vertex_Program;
    
    public:    
        Matrix(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, bool directed_, bool transpose_, bool parallel_edges_,
               Tiling_type tiling_type_, Compression_type compression_type_);
        ~Matrix();
        
    private:
        Integer_Type nrows, ncols;
        uint32_t ntiles, nrowgrps, ncolgrps;
        Integer_Type tile_height, tile_width;    
        uint64_t nedges = 0;
        Tiling* tiling;
        Compression_type compression_type;
        Filtering_type filtering_type;
        bool directed;
        bool transpose;
        bool parallel_edges;

        
        std::vector<char> II;           // Nonzero rows bitvectors (from tile width)
        std::vector<Integer_Type> IIV;  // Nonzero rows indices    (from tile width)
        std::vector<char> JJ;           // Nonzero cols bitvectors (from tile height)
        std::vector<Integer_Type> JJV;  // Nonzero cols indices    (from tile height)
        std::vector<Integer_Type> rows_sizes;
        Integer_Type rows_size;
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_rows_values;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        Integer_Type nnz_cols_size_loc;
        std::vector<Integer_Type> start_dense;
        std::vector<Integer_Type> end_dense;
        std::vector<Integer_Type> start_sparse;
        std::vector<Integer_Type> end_sparse;        
        
        std::vector<std::vector<char>> I;           // Nonzero rows bitvectors (from tile width)
        std::vector<std::vector<Integer_Type>> IV;  // Nonzero rows indices    (from tile width)
        std::vector<std::vector<char>> J;           // Nonzero cols bitvectors (from tile height)
        std::vector<std::vector<Integer_Type>> JV;  // Nonzero cols indices    (from tile height)
        std::vector<std::vector<char>> IT;           // Nonzero rows bitvectors (from tile width)
        std::vector<std::vector<Integer_Type>> IVT;  // Nonzero rows indices    (from tile width)
        std::vector<Integer_Type> threads_nnz_rows;  // Row group row indices
        std::vector<Integer_Type> threads_start_row;  
        std::vector<Integer_Type> threads_nnz_start_row;  
        std::vector<Integer_Type> threads_end_row;  
        std::vector<Integer_Type> rowgrp_nnz_rows;  // Row group row indices         
        std::vector<Integer_Type> rowgrp_regular_rows; // Row group regular indices
        std::vector<Integer_Type> rowgrp_source_rows;  // Row group source column indices
        std::vector<Integer_Type> colgrp_nnz_columns;  // Column group column indices
        std::vector<Integer_Type> colgrp_sink_columns;    // Column group sink column indices
        std::vector<std::vector<Integer_Type>> regular_rows;// Row group regular indices
        std::vector<std::vector<char>> regular_rows_bitvector;// Row group regular bitvector
        std::vector<std::vector<Integer_Type>> source_rows;// Row group source indices
        std::vector<std::vector<char>> source_rows_bitvector;// Row group source bitvector
        std::vector<std::vector<Integer_Type>> regular_columns;// Col group regular indices
        std::vector<std::vector<char>> regular_columns_bitvector;// Col group regular bitvector
        std::vector<std::vector<Integer_Type>> sink_columns;// Col group sink indices
        std::vector<std::vector<char>> sink_columns_bitvector;// Col group sink bitvector
        
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles;
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles_rg;
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles_cg;
        
        std::vector<uint32_t> local_tiles;
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;
        std::vector<int32_t> local_row_segments;
        std::vector<int32_t> local_col_segments;
        
        std::vector<int32_t> leader_ranks;
        
        std::vector<int32_t> leader_ranks_rg;
        std::vector<int32_t> leader_ranks_cg;
        
        std::vector<int32_t> all_rowgrp_ranks;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        std::vector<int32_t> all_colgrp_ranks; 
        std::vector<int32_t> all_colgrp_ranks_accu_seg;
        
        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        std::vector<int32_t> follower_colgrp_ranks; 
        std::vector<int32_t> follower_colgrp_ranks_accu_seg;
        
        
        std::vector<int32_t> all_rowgrp_ranks_rg;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> all_colgrp_ranks_cg;
        std::vector<int32_t> all_colgrp_ranks_accu_seg_cg;
        
        std::vector<int32_t> follower_rowgrp_ranks_rg;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> follower_colgrp_ranks_cg;
        std::vector<int32_t> follower_colgrp_ranks_accu_seg_cg;
        int32_t owned_segment, accu_segment_rg, accu_segment_cg, accu_segment_row, accu_segment_col;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        
        
        void free_tiling();
        void init_matrix();
        void del_triples();
        void del_triples_t();
        void init_tiles();
        void init_threads();
        void closest(std::vector<struct Triple<Weight, Integer_Type>>& triples, int32_t npartitions, uint64_t chunk_size_, 
                     std::vector<uint64_t>& start, std::vector<uint64_t>& end);
        void init_compression();
        //void init_csr();
        void init_csc();
        //void init_tcsr();
        void init_tcsc();
        void init_tcsc_cf();
        //void init_dcsr();
        void init_dcsc();
        //void init_bv();
        //void del_csr();
        //void del_csc();
        //void del_dcsr();
        //void construct_filter();
        //void destruct_filter();
        void del_compression();
        void del_filter();
        void del_classifier();
        //void del_filtering_indices();
        void print(std::string element);
        void distribute();
        void init_filtering();
        void filter_vertices(Filtering_type filtering_type_);
        void classify_vertices();
        void filter_rows();
        void filter_cols();
        
        
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type>& triple);
        uint32_t local_tile_of_triple(const struct Triple<Weight, Integer_Type>& triple);
        uint32_t segment_of_tile(const struct Triple<Weight, Integer_Type>& pair);
		uint32_t owner_of_tile(const struct Triple<Weight, Integer_Type>& pair);
        uint32_t row_leader_of_tile(const struct Triple<Weight, Integer_Type>& pair);
		
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type>& pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type>& pair);
        void insert(const struct Triple<Weight, Integer_Type>& triple);
        void test(const struct Triple<Weight, Integer_Type>& triple);      
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, bool directed_, bool transpose_, bool parallel_edges_, Tiling_type tiling_type_, 
    Compression_type compression_type_) {
    nrows = nrows_;
    ncols = ncols_;
    ntiles = ntiles_;
    if((tiling_type_ == _2D_) or (tiling_type_ == _2DT_)){
        nrowgrps = sqrt(ntiles);
        ncolgrps = ntiles / nrowgrps;
        tile_height = (nrows / nrowgrps) + 1;
        tile_width  = (ncols / ncolgrps) + 1;
    }
    else {
        nrowgrps = 1;
        ncolgrps = ntiles;
        tile_height = nrows;
        tile_width  = ncols / ncolgrps;
        //printf("rank=%d tile_height=%d tile_width=%d\n", Env::rank, tile_height, tile_width);
        
        //if(Env::rank == (Env::nranks - 1))
        //    tile_width = ncols  - ((Env::nranks - 1) * (ncols / ncolgrps));
    }
    
    
    directed = directed_;
    transpose = transpose_;
    parallel_edges = parallel_edges_;
    // Initialize tiling 
    tiling = new Tiling(Env::nranks, ntiles, nrowgrps, ncolgrps, tiling_type_);
    compression_type = compression_type_;
    init_matrix();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::~Matrix(){};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::free_tiling() {
    delete tiling;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::tile_of_local_tile(const uint32_t local_tile) {
    return{(local_tile - (local_tile % ncolgrps)) / ncolgrps, local_tile % ncolgrps};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::tile_of_triple(const struct Triple<Weight, Integer_Type>& triple) {
    return{(triple.row / tile_height), (triple.col / tile_width)};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::segment_of_tile(const struct Triple<Weight, Integer_Type>& pair) {
    return(pair.col);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::owner_of_tile(const struct Triple<Weight, Integer_Type>& pair) {
    return(tiles[pair.row][pair.col].rank);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::row_leader_of_tile(const struct Triple<Weight, Integer_Type>& pair) {
    return(tiles[pair.row][pair.row].rank);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::base(const struct Triple<Weight, Integer_Type>& pair, 
                      Integer_Type rowgrp, Integer_Type colgrp) {
   return{(pair.row + (rowgrp * tile_height)), (pair.col + (colgrp * tile_width))};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Triple<Weight, Integer_Type> Matrix<Weight, Integer_Type, Fractional_Type>::rebase(const struct Triple<Weight, Integer_Type>& pair) {
    return{(pair.row % tile_height), (pair.col % tile_width)};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::test(const struct Triple<Weight, Integer_Type>& triple) {        
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    //uint32_t t = (pair.row * tiling->ncolgrps) + pair.col;
    //if(not (std::find(local_tiles.begin(), local_tiles.end(), t) != local_tiles.end())) {
    if(tiles[pair.row][pair.col].rank != Env::rank) {
        printf("Rank=%d: Invalid entry for tile[%d][%d]=[%d %d]\n", Env::rank, pair.row, pair.col, triple.row, triple.col);
        Env::exit(1);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
uint32_t Matrix<Weight, Integer_Type, Fractional_Type>::local_tile_of_triple(const struct Triple<Weight, Integer_Type>& triple) {
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    uint32_t t = (pair.row * tiling->ncolgrps) + pair.col;
    return(t);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::insert(const struct Triple<Weight, Integer_Type>& triple) {
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    tiles[pair.row][pair.col].triples->push_back(triple);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_matrix() {
    // Reserve the 2D vector of tiles. 
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++)
        tiles[i].resize(ncolgrps);
    // Initialize tiles 
    
    
    
    
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            tile.rg = i;
            tile.cg = j;
            if(tiling->tiling_type == Tiling_type::_2D_) {
                tile.rank = ((j % tiling->rowgrp_nranks) * tiling->colgrp_nranks) +
                                   (i % tiling->colgrp_nranks);
                tile.ith = tile.rg   / tiling->colgrp_nranks;
                tile.jth = tile.cg   / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = i;
                tile.leader_rank_cg = j;
                
                tile.leader_rank_rg_rg = i;
                tile.leader_rank_cg_cg = j;
            }
            else if(tiling->tiling_type == Tiling_type::_2DT_) {
                
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks
                                                        + (j % tiling->rowgrp_nranks);
                
                tile.ith = tile.rg / tiling->colgrp_nranks; 
                tile.jth = tile.cg / tiling->rowgrp_nranks;
                
                tile.rank_rg = j % tiling->rowgrp_nranks;
                tile.rank_cg = i % tiling->colgrp_nranks;
                
                tile.leader_rank_rg = i;
                tile.leader_rank_cg = j;
                
                tile.leader_rank_rg_rg = i;
                tile.leader_rank_cg_cg = j;
            }
            else if(tiling->tiling_type == Tiling_type::_1D_COL_) {
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks
                                                        + (j % tiling->rowgrp_nranks);
                
                tile.ith = tile.rg / tiling->colgrp_nranks; 
                tile.jth = tile.cg / tiling->rowgrp_nranks;
                
            }
            
            tile.nth   = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.mth   = (tile.jth * tiling->rank_nrowgrps) + tile.ith;
            tile.allocate_triples();
        }
    }

    if((tiling->tiling_type == Tiling_type::_2D_) or (tiling->tiling_type == Tiling_type::_2DT_)) {
        /*
        * Reorganize the tiles so that each rank is placed in
        * at least one diagonal tile then calculate 
        * the leader ranks per row group.
        */
        leader_ranks.resize(nrowgrps, -1);
        leader_ranks_rg.resize(nrowgrps);
        leader_ranks_cg.resize(ncolgrps);
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = i; j < ncolgrps; j++) {
                if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[j][i].rank)
                     != leader_ranks.end())) {
                    std::swap(tiles[j], tiles[i]);
                    break;
                }
            }
            leader_ranks[i] = tiles[i][i].rank;
            leader_ranks_rg[i] = tiles[i][i].rank_rg;
            leader_ranks_cg[i] = tiles[i][i].rank_cg;
        }
        
        //Calculate local tiles and local column segments
        struct Triple<Weight, Integer_Type> pair;
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                tile.rg = i;
                tile.cg = j;
                tile.kth   = (tile.rg * tiling->ncolgrps) + tile.cg;
                if(tile.rank == Env::rank) {
                    pair.row = i;
                    pair.col = j;    
                    local_tiles.push_back(tile.kth);
                    local_tiles_row_order.push_back(tile.kth);
                    if (std::find(local_col_segments.begin(), local_col_segments.end(), pair.col) == local_col_segments.end())
                        local_col_segments.push_back(pair.col);
                    
                    if (std::find(local_row_segments.begin(), local_row_segments.end(), pair.row) == local_row_segments.end())
                        local_row_segments.push_back(pair.row);
                }
                tile.leader_rank_rg = tiles[i][i].rank;
                tile.leader_rank_cg = tiles[j][j].rank;
                    
                tile.leader_rank_rg_rg = tiles[i][i].rank_rg;
                tile.leader_rank_cg_cg = tiles[j][j].rank_cg;
                if((tile.rank == Env::rank) and (i == j)) {
                    owned_segment = i;
                    //owned_segment_vec.push_back(owned_segment);
                }
            }
        }
        
        for (uint32_t j = 0; j < ncolgrps; j++) {
            for (uint32_t i = 0; i < nrowgrps; i++) {
                auto& tile = tiles[i][j];
                if(tile.rank == Env::rank)
                    local_tiles_col_order.push_back(tile.kth);
            }
        }
        
        for(uint32_t t: local_tiles_row_order) {
            pair = tile_of_local_tile(t);
            if(pair.row == pair.col) {            
                for(uint32_t j = 0; j < ncolgrps; j++) {
                    if(tiles[pair.row][j].rank == Env::rank) {
                        if(std::find(all_rowgrp_ranks.begin(), all_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                                  == all_rowgrp_ranks.end()) {
                            all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                            all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                    
                            all_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                            all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                        }
                    }
                    else {
                        if(std::find(follower_rowgrp_ranks.begin(), follower_rowgrp_ranks.end(), tiles[pair.row][j].rank) 
                                == follower_rowgrp_ranks.end()) {
                            all_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                            all_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                            
                            all_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                            all_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                            
                            follower_rowgrp_ranks.push_back(tiles[pair.row][j].rank);
                            follower_rowgrp_ranks_accu_seg.push_back(tiles[pair.row][j].cg);
                            
                            follower_rowgrp_ranks_rg.push_back(tiles[pair.row][j].rank_rg);
                            follower_rowgrp_ranks_accu_seg_rg.push_back(tiles[pair.row][j].cg);
                        }
                    }
                }
                for(uint32_t i = 0; i < nrowgrps; i++) {
                    if(tiles[i][pair.col].rank == Env::rank) {
                        if(std::find(all_colgrp_ranks.begin(), all_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                                  == all_colgrp_ranks.end()) {
                            all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                            all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                
                            all_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                            all_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                        }
                    }
                    else {
                        if(std::find(follower_colgrp_ranks.begin(), follower_colgrp_ranks.end(), tiles[i][pair.col].rank) 
                                  == follower_colgrp_ranks.end()) {
                            all_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                            all_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                            
                            all_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                            all_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                            
                            follower_colgrp_ranks.push_back(tiles[i][pair.col].rank);
                            follower_colgrp_ranks_accu_seg.push_back(tiles[i][pair.col].rg);
                            
                            follower_colgrp_ranks_cg.push_back(tiles[i][pair.col].rank_cg);
                            follower_colgrp_ranks_accu_seg_cg.push_back(tiles[i][pair.col].rg);
                        }
                    }
                }
                break;
                /* We do not keep iterating as the ranks in row/col groups are the same */
            }
        }
        
        // Spilitting communicator among row/col groups       
        indexed_sort<int32_t, int32_t>(all_rowgrp_ranks, all_rowgrp_ranks_accu_seg);
        indexed_sort<int32_t, int32_t>(all_rowgrp_ranks_rg, all_rowgrp_ranks_accu_seg_rg);
        // Make sure there is at least one follower
        if(follower_rowgrp_ranks.size() > 1) {
            indexed_sort<int32_t, int32_t>(follower_rowgrp_ranks, follower_rowgrp_ranks_accu_seg);
            indexed_sort<int32_t, int32_t>(follower_rowgrp_ranks_rg, follower_rowgrp_ranks_accu_seg_rg);
        }
        indexed_sort<int32_t, int32_t>(all_colgrp_ranks, all_colgrp_ranks_accu_seg);
        indexed_sort<int32_t, int32_t>(all_colgrp_ranks_cg, all_colgrp_ranks_accu_seg_cg);
        // Make sure there is at least one follower
        if(follower_colgrp_ranks.size() > 1) {
            indexed_sort<int32_t, int32_t>(follower_colgrp_ranks, follower_colgrp_ranks_accu_seg);
            indexed_sort<int32_t, int32_t>(follower_colgrp_ranks_cg, follower_colgrp_ranks_accu_seg_cg);
        }
        if(Env::comm_split and not Env::get_comm_split()) {
            Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
            Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
            Env::set_comm_split();
        }
        // Which column index in my rowgrps is mine when I'm the accumulator
        for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++) {
            if(all_rowgrp_ranks[j] == Env::rank)
                accu_segment_rg = j;
        }
        // Which row index in my colgrps is mine when I'm the accumulator
        for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++) {
            if(all_colgrp_ranks[j] == Env::rank)
                accu_segment_cg = j;
        } 
        // Which rowgrp is mine
        for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++) {
            if(leader_ranks[local_row_segments[j]] == Env::rank)
                accu_segment_row = j;
        }
        // Which colgrp is mine
        for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++) {
            if(leader_ranks[local_col_segments[j]] == Env::rank)
                accu_segment_col = j;
        } 
    } 
    else if(tiling->tiling_type == Tiling_type::_1D_COL_) {
        accu_segment_row = Env::rank;
        //follower_rowgrp_ranks.resize(Env::nranks - 1);
        for(uint32_t j = 0; j < tiling->rowgrp_nranks; j++) {
            if(j != (uint32_t) Env::rank)
                follower_rowgrp_ranks.push_back(j);
        }
    }
    
    
    // Print tiling assignment
    if(Env::is_master) {
        printf("Tiling Info: %d x %d [nrows x ncols]\n", nrows, ncols);
        printf("Tiling Info: %d x %d [rowgrps x colgrps]\n", nrowgrps, ncolgrps);
        printf("Tiling Info: %d x %d [height x width]\n", tile_height, tile_width);
        printf("Tiling Info: %d x %d [rowgrp_nranks x colgrp_nranks]\n", tiling->rowgrp_nranks, tiling->colgrp_nranks);
        printf("Tiling Info: %d x %d [rank_nrowgrps x rank_ncolgrps]\n", tiling->rank_nrowgrps, tiling->rank_ncolgrps);
    }
    print("rank");
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::print(std::string element)
{
    if(Env::is_master) {    
        uint32_t skip = 15;
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                if(element.compare("rank") == 0) 
                    printf("%02d ", tile.rank);
                else if(element.compare("kth") == 0) 
                    printf("%3d ", tile.kth);
                else if(element.compare("ith") == 0) 
                    printf("%2d ", tile.ith);
                else if(element.compare("jth") == 0) 
                    printf("%2d ", tile.jth);
                else if(element.compare("nth") == 0) 
                    printf("%2d ", tile.jth);
                else if(element.compare("mth") == 0) 
                    printf("%2d ", tile.jth);
                else if(element.compare("rank_rg") == 0) 
                    printf("%2d ", tile.rank_rg);
                else if(element.compare("rank_cg") == 0) 
                    printf("%2d ", tile.rank_cg);
                else if(element.compare("leader_rank_rg") == 0) 
                    printf("%2d ", tile.leader_rank_rg);
                else if(element.compare("leader_rank_cg") == 0) 
                    printf("%2d ", tile.leader_rank_cg);
                else if(element.compare("nedges") == 0) 
                    printf("%lu ", tile.nedges);
                if(j > skip) {
                    printf("...");
                    break;
                }
            }
            printf("\n");
            if(i > skip) {
                printf(".\n.\n.\n");
                break;
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tiles() {
    
    
    /*
    if(Env::is_master) {
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
                printf( "%d ", triples.size());
            }
            printf( "\n");
        }
    }
    */
    
    MPI_Barrier(MPI_COMM_WORLD);
    distribute();
    MPI_Barrier(MPI_COMM_WORLD);
    
    //Triple<Weight, Integer_Type> pair;
    RowSort<Weight, Integer_Type> f_row;
    auto f_comp = [] (const Triple<Weight, Integer_Type> &a, const Triple<Weight, Integer_Type> &b) {return (a.row == b.row and a.col == b.col);};    
    
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
    //for (uint32_t i = 0; i < nrowgrps; i++) {
      //  for (uint32_t j = 0; j < ncolgrps; j++) {
            //auto& tile = tiles[i][j];
            //if(tile.rank == Env::rank) {
                std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
                if(triples.size()) {
                    std::sort(triples.begin(), triples.end(), f_row);
                    /* remove parallel edges (duplicates), necessary for triangle couting */
                    if(not parallel_edges) {
                        auto last = std::unique(triples.begin(), triples.end(), f_comp);
                        triples.erase(last, triples.end());
                    }
                }
                tile.nedges = tile.triples->size();
                //printf("%d %d\n", Env::rank, tile.nedges);
            //}
        }
    //}
    
    
    
    /*
    std::vector<uint64_t> nnz_global(Env::nranks);
    nnz_global[Env::rank] = tiles[0][Env::rank].triples->size();
    for (int32_t i = 0; i < Env::nranks; i++) {
        if (i != Env::rank) {
            MPI_Sendrecv(&nnz_global[Env::rank], 1, MPI_UNSIGNED_LONG, i, 0, &nnz_global[i], 1, MPI_UNSIGNED_LONG,
                                                        i, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if(!Env::rank) {
        double sum = std::accumulate(nnz_global.begin(), nnz_global.end(), 0.0);
        double mean = sum / Env::nranks;
        double sq_sum = std::inner_product(nnz_global.begin(), nnz_global.end(), nnz_global.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / Env::nranks - mean * mean);        
        printf("Edge distribution: MPI edges  (sum: avg +/- std_dev)= %.0f: %.0f +/- %.0f\n", sum, mean, std_dev);
    }

    */
    
    
}


/* Inspired from LA3 code @
   https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/dist_matrix2d.hpp
*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::distribute()
{
    if(Env::is_master)
        printf("Edge distribution: Distributing edges among %d ranks\n", Env::nranks);     
    
    
   /* Sanity check on # of edges */
    uint64_t nedges_start_local = 0, nedges_end_local = 0,
             nedges_start_global = 0, nedges_end_global = 0;
             
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;                 
             
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.triples->size() > 0)
                nedges_start_local += tile.triples->size();
        }
    }

    MPI_Datatype MANY_TRIPLES;
    const uint32_t many_triples_size = 1;
    MPI_Type_contiguous(many_triples_size * sizeof(Triple<Weight, Integer_Type>), MPI_BYTE, &MANY_TRIPLES);
    MPI_Type_commit(&MANY_TRIPLES);
    std::vector<std::vector<Triple<Weight, Integer_Type>>> outboxes(Env::nranks);
    std::vector<std::vector<Triple<Weight, Integer_Type>>> inboxes(Env::nranks);
    std::vector<uint32_t> inbox_sizes(Env::nranks);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++)   {
            auto& tile = tiles[i][j];
            if(tile.rank != Env::rank) {
                auto& outbox = outboxes[tile.rank];
                outbox.insert(outbox.end(), tile.triples->begin(), tile.triples->end());
                tile.free_triples();
            }
        }
    }
  
    for (int32_t r = 0; r < Env::nranks; r++) {
    //for (int32_t i = 0; i < Env::nranks; i++) {
        //int32_t r = (Env::rank + i) % Env::nranks;
        if (r != Env::rank) {
            auto& outbox = outboxes[r];
            uint32_t outbox_size = outbox.size();
            MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, 0, &inbox_sizes[r], 1, MPI_UNSIGNED,
                                                        r, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
            auto &inbox = inboxes[r];
            inbox.resize(inbox_sizes[r]);
            MPI_Sendrecv(outbox.data(), outbox.size(), MANY_TRIPLES, r, 0, inbox.data(), inbox.size(), MANY_TRIPLES,
                                                        r, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  

/*
    MPI_Barrier(MPI_COMM_WORLD);    
    for (int32_t r = 0; r < Env::nranks; r++) {
        if (r != Env::rank) {
            auto &outbox = outboxes[r];
            uint32_t outbox_size = outbox.size();
            MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, Env::rank, &inbox_sizes[r], 1, MPI_UNSIGNED, 
                                                        r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    
    
    for (int32_t r = 0; r < Env::nranks; r++) {
        if (r != Env::rank) {
            auto &inbox = inboxes[r];
            inbox.resize(inbox_sizes[r]);
            auto &outbox = outboxes[r];
            MPI_Sendrecv(outbox.data(), outbox.size(), MANY_TRIPLES, r, Env::rank, inbox.data(), inbox.size(), MANY_TRIPLES, 
                                                        r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
   */ 
    
    /*
    for (int32_t i = 0; i < Env::nranks; i++) {
        int32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank) {
            auto &inbox = inboxes[r];
            inbox.resize(inbox_sizes[r]);
            MPI_Irecv(inbox.data(), inbox.size(), MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);    
            in_requests.push_back(request);
        }
    }
    for (int32_t i = 0; i < Env::nranks; i++) {
        int32_t r = (Env::rank + i) % Env::nranks;
        if(r != Env::rank) {
            auto &outbox = outboxes[r];
            MPI_Isend(outbox.data(), outbox.size(), MANY_TRIPLES, r, 1, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }
    }     
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);   
    out_requests.clear();
    
    */

    for (int32_t r = 0; r < Env::nranks; r++) {
        if (r != Env::rank) {
            auto& inbox = inboxes[r];
            for (uint32_t i = 0; i < inbox_sizes[r]; i++) {
                test(inbox[i]);
                insert(inbox[i]);
            }
            inbox.clear();
            inbox.shrink_to_fit();
        }
    }

    
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) {
                std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
                nedges_end_local += triples.size();
            }
        }
    }
    
    
    /*
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        nedges_end_local += tile.triples->size();
    }
    */
    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    MPI_Allreduce(&nedges_end_local, &nedges_end_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges_start_global == nedges_end_global);
    if(Env::is_master)
        printf("Edge distribution: Sanity check for exchanging %lu edges is done\n", nedges_end_global);
    auto retval = MPI_Type_free(&MANY_TRIPLES);
    assert(retval == MPI_SUCCESS);   
    Env::barrier();
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::closest(std::vector<struct Triple<Weight, Integer_Type>>& triples,
                   int32_t npartitions, uint64_t chunk_size_, std::vector<uint64_t>& start, std::vector<uint64_t>& end) {
                       
   // printf("size=%lu chunk_size=%lu, npartitions=%d\n", triples.size(), chunk_size, npartitions);
    uint64_t chunk_size = chunk_size_;
    uint64_t m = triples.size();
    uint32_t k = 0;
    //start[0] = 0;
    //end[npartitions - 1] = m;
    for(int i = 0; i < npartitions; i++) {
        if(i == 0)
            start[i] = 0;
        else
            start[i] = end[i-1] + 1;
        
        bool fl = false;
        bool fr = false;
        bool f = false;
        k = (i + 1) * chunk_size;
        
        
/*        
        if(i > 0) {
            if(jl > (k + chunk_size) or jr > (k + chunk_size)) {
                break;
            }

        }
        */

        uint32_t jl = k - 1;
        uint32_t jr = k;
        uint32_t r = triples[k].row;
        
        while(jl >= start[i]) {
            if(r != triples[jl].row) {
                //end[i] = jl;
                fl = true;
                //jl++;
                break;
            }
            jl--;
        }
        
        while(jr < m) {
            if(r != triples[jr].row) {
                //end[i] = jr;
                fr = true;
                jr--;
                break;
            }
            jr++;
        }
        
        if(i < npartitions - 1) {
            if(fl and fr){
                if((k - jl) <= (k - jr)) {
                    //if(jl > start[i])
                        end[i] = jl;
                    //else
                        //end[i] = jr;
                }
                else {
                   
                    end[i] = jr;
                }
            }
            else if(fl) 
                end[i] = jl;
            else if(fr) 
                end[i] = jr;
            else
                end[i] = start[i];
        }
        else 
            end[i] = m - 1;
        
         
       
        
        //printf("%d %d %d %d %d [%d %d %d %d]\n", i, start[i], end[i], jr, jl,  triples[jl].row, r, triples[jr].row, triples[jr-1].row);    
        
        
        /*
        //if(triples[idx].row == triples[idx+1].row) {
        while((jl >= start[i]) and (jr < m)) {
            if(r != triples[jl].row) {
                fl = true;
            }
            if(r != triples[jr].row) {
                fr = true;
            }
            if(fl or fr)
                break;
            jl--;
            jr++;
        }
        
        if(fl and fr) {
            if((k - jl) <= (k - jr))
                end[i] = jl;
            else
                end[i] = jr;
        }
        else if(fl) {
            end[i] = jl;
        }
        else if(fr) {
            end[i] = jr;
        } else {
            if(jl >= start[i]) {
                while(jl >= start[i]) {
                    if(r != triples[jl].row) {
                        end[i] = jl;
                        f = true;
                        break;
                    }
                    jl--;
                }
            }
            else {
                while(jr < m) {
                    if(r != triples[jr].row) {
                        end[i] = jr;
                        f = true;
                        break;
                    }
                    jr++;
                }
            }
        }
        
        if(not(fl or fr)) {
            if(not f)
                end[i] = start[i];
        }
        if(i == npartitions - 1)
            end[i] = m;
        */
        /*
        if(!Env::rank) {
        printf("i=%d [start=%lu end=%lu]: length=%lu [ch_st=%lu ch_end=%lu] left=%d right=%d k=%d [%d %d]\n", i, start[i], end[i], end[i] - start[i], i* chunk_size, (i + 1) * chunk_size, jl, jr, k, fl ,fr);
        if(i == 0)
            printf("i=%d [%d|%d, %d|%d|%d]\n", i, triples[start[i]].row, triples[start[i]+1].row, triples[end[i]-1].row, triples[end[i]].row, triples[end[i]+1].row);
        else if( i > 0 and i < npartitions - 1)
            printf("i=%d [%d|%d|%d, %d|%d|%d]\n", i, triples[start[i]-1].row, triples[start[i]].row, triples[start[i]+1].row, triples[end[i]-1].row, triples[end[i]].row, triples[end[i]+1].row);
        else
            printf("i=%d [%d|%d|%d, %d|%d]\n", i, triples[start[i]-1].row, triples[start[i]].row, triples[start[i]+1].row, triples[end[i]-1].row, triples[end[i]].row);
        }
        */
        /*
        if(i > 0) {
            if(jl > (k + chunk_size) or jr > (k + chunk_size)) {
                break;
            }

        }
        */
        
    }
    

    
    for(int32_t i = 1; i < npartitions; i++) {
        if(triples[end[i-1]].row == triples[start[i]].row)
            assert(triples[start[i-1]].row == triples[end[i]].row);
    }
        //if(triples[start[i-1]].row > triples[end[i]].row) {
          //  printf("i=%d %d %d\n", i, triples[start[i-1]].row, triples[end[i]].row);
        //}
    
    

    
    //Env::barrier();
    //std::exit(0);
/*
    int jl = q - 1;
    int jr = q + 1;
    while((jl >= 0) and (jr < n)) {
        if(c == s[jl]) {
            fl = true;
        }
        if(c == s[jr]) {
            fr = true;
        }
        if(fl or fr)
            break;
        jl--;
        jr++;
    }
    
    if(fl and fr) {
        if((q - jl) <= (jr - q))
            output[i] = jl;
        else
            output[i] = jr;
    }
    else if(fl) {
        output[i] = jl;
    }
    else if(fr) {
        output[i] = jr;
    } else {
        if(jl >= 0) {
            while(jl >= 0) {
                if(c == s[jl]) {
                    output[i] = jl;
                    break;
                }
                jl--;
            }
        }
        else {
            while(jr < n) {
                if(c == s[jr]) {
                    output[i] = jr;
                    break;
                }
                jr++;
            }
        }
    }
    */
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_threads() {
    if(Env::is_master)
        printf("Edge distribution: Distributing edges among %d threads\n", omp_get_max_threads());     
    ColSort<Weight, Integer_Type> f_col;
    
    //for (uint32_t i = 0; i < nrowgrps; i++) {
      //  for (uint32_t j = 0; j < ncolgrps; j++) {
        //    auto& tile = tiles[i][j];
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];        
        
        
            //if(tile.rank == Env::rank) {
                std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
                if(triples.size()) {
                    tile.npartitions = omp_get_max_threads();
                    threads_start_row.resize(tile.npartitions);
                    threads_end_row.resize(tile.npartitions);
                    tile.triples_t.resize(tile.npartitions); 
                    uint64_t chunk_size = tile.triples->size()/omp_get_max_threads();
                    std::vector<uint64_t> start(tile.npartitions);
                    std::vector<uint64_t> end(tile.npartitions);
                    closest(triples, tile.npartitions, chunk_size, start, end);
                    
                    
                    std::vector<uint64_t> nnz_local(tile.npartitions);
                    #pragma omp parallel
                    {
                        int tid = omp_get_thread_num();
                        tile.triples_t[tid] = new std::vector<struct Triple<Weight, Integer_Type>>;
                        for(uint64_t i = start[tid]; i <= end[tid]; i++) {
                            auto& triple = triples[i];
                            tile.triples_t[tid]->push_back(triple); 
                        }
                        std::sort(tile.triples_t[tid]->begin(), tile.triples_t[tid]->end(), f_col);
                        nnz_local[tid] = tile.triples_t[tid]->size();

                        

                        
                        //threads_start_row[tid] = (tid != 0) ? triples[end[tid + 1]].row : 0;
                        //threads_end_row[tid] = triples[end[tid]].row;
                        //threads_end_row[tid] = (threads_end_row[tid] < threads_start_row[tid]) ? threads_start_row[tid] : threads_end_row[tid];
                        //threads_end_row[tid] = (tid == Env::nthreads - 1) ? tile_height : threads_end_row[tid] + 1;
                        
                    }
                    
                    double sum = std::accumulate(nnz_local.begin(), nnz_local.end(), 0.0);
                    double mean = sum / tile.npartitions;
                    double sq_sum = std::inner_product(nnz_local.begin(), nnz_local.end(), nnz_local.begin(), 0.0);
                    double std_dev = std::sqrt(sq_sum / tile.npartitions - mean * mean);
                    if(!Env::rank)
                        printf("Edge distribution: Rank %d tile %d - Threads edges (sum: avg +/- std_dev)= %.0f: %.0f +/- %.0f\n", Env::rank, t, sum, mean, std_dev);
                    
                    /*
                    for(int i = 0; i < tile.npartitions; i++) {
                        
                        if(i == tile.npartitions - 1)
                            threads_end_row[i] = tile_height;
                        else 
                            threads_end_row[i] = triples[end[i]].row + 1;
                        
                        if(i == 0)
                            threads_start_row[i] = 0;
                        else {
                            threads_start_row[i] = threads_end_row[i - 1];
                        }
                        //if(!Env::rank)
                       // printf("rank=%d thread=%d size=%lu start=%d end=%d\n", Env::rank, i, tile.triples_t[i]->size(), threads_start_row[i], threads_end_row[i]);
                    }
                    */
                    
                }
            //}
        //}
    }
   del_triples();


   
    /*
    if(!Env::rank) {
        auto& tile = tiles[0][Env::rank];
        for(int i = 0; i < Env::nthreads; i++) {
            std::vector<struct Triple<Weight, Integer_Type>>& triples_t = *(tile.triples_t[i]);
            struct Triple<Weight, Integer_Type> first = triples_t.front();
            struct Triple<Weight, Integer_Type> last = triples_t.back();
            printf("%d %lu %d %d\n", i, triples_t.size(), first.row, last.row);
        }
    }
    
    
    Env::barrier();
    Env::exit(0);    
    */
    
    /*

    
    Triple<Weight, Integer_Type> pair;
    
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
        //int c = 0;
        if(triples.size()) {
            tile.npartitions = omp_get_max_threads();
            tile.triples_t.resize(tile.npartitions); 
            uint64_t chunk_size = tile.triples->size()/omp_get_max_threads();
            std::vector<uint64_t> start(tile.npartitions);
            std::vector<uint64_t> end(tile.npartitions);
            closest(triples, tile.npartitions, chunk_size, start, end);
            */
            /*
            uint64_t start_idx = 0;
            uint64_t end_idx = 0;
            uint64_t idx = 0;
            uint64_t idx1 = 0;
            bool sentinel = false;
            
            printf("size=%lu chunk_size=%lu, %d\n", tile.triples->size(), chunk_size, tile.npartitions);
            for(int i = 0; i < tile.npartitions; i++) {
                if(i == 0)
                    start_idx = 0;
                else
                    start_idx = end_idx;
                
                if(i < (tile.npartitions - 1)) {
                    
                    //idx += chunk_size;
                    if(idx <= ((i + 1) * chunk_size)) {
                        idx = (i + 1) * chunk_size;
                    }
                    else 
                        idx += chunk_size;
                    
                    printf("i=%d idx=%d\n", i, idx);
                    
                    while(true) {
                        if(idx+1 > triples.size()) {
                            sentinel = true;
                            break;
                        }
                        
                        //    printf("ERROR: i=%d idx=%d\n", i, idx);
                        //assert(idx+1 < triples.size());
                        if(triples[idx].row == triples[idx+1].row) {
                            idx++;
                            c++;
                        }
                        else
                            break;
                        
                    }
                    
                    if(not sentinel)
                        end_idx = idx + 1;
                    else
                        end_idx = triples.size();
                }
                else
                    end_idx = triples.size();
                
                start[i] = start_idx;
                end[i] = end_idx;
                //avg_nnz += (end[i] - start[i]);
                printf("%d [%lu %lu]: %lu [%lu %lu] %d\n", i, start[i], end[i], end[i] - start[i], i* chunk_size, (i + 1) * chunk_size, c);
                c = 0;
            }
            */
            /*
            std::vector<uint64_t> nnz_local(tile.npartitions);
            //uint64_t nnz_global = 0;
            #pragma omp parallel //private(nnz_local) reduction(+:nnz_global)
            {
                int tid = omp_get_thread_num();
                tile.triples_t[tid] = new std::vector<struct Triple<Weight, Integer_Type>>;
                for(uint64_t i = start[tid]; i < end[tid]; i++) {
                    auto& triple = triples[i];
                    tile.triples_t[tid]->push_back(triple); 
                }
                std::sort(tile.triples_t[tid]->begin(), tile.triples_t[tid]->end(), f_col);
                nnz_local[tid] = tile.triples_t[tid]->size();
                //nnz_global += nnz_local;
                //printf("%d %lu\n", tid, tile.triples_t[tid]->size());
            }
            
            
            
            double sum = std::accumulate(nnz_local.begin(), nnz_local.end(), 0.0);
            double mean = sum / tile.npartitions;
            double sq_sum = std::inner_product(nnz_local.begin(), nnz_local.end(), nnz_local.begin(), 0.0);
            double std_dev = std::sqrt(sq_sum / tile.npartitions - mean * mean);
            if(!Env::rank)
                printf("Edge distribution: Rank %d tile %d - Threads edges (sum: avg +/- std_dev)= %f: %f +/- %f\n", Env::rank, t, sum, mean, std_dev);
            
        }
    }
    */
//    
}




template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_filtering() {
    
    if(Env::is_master)
        printf("Vertex filtering: Filtering zero rows and columns\n");
    
    
    /*
    filter_rows();

    
    filter_cols();

    */    

    
    //printf("[x]init_filtering()\n");
    
    if(Env::is_master)
        printf("Vertex filtering: Filtering nonzero rows\n");    
    I.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        I[i].resize(tile_height);
    IV.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        IV[i].resize(tile_height);
    filter_vertices(_ROWS_);
    uint32_t io = accu_segment_row;
    auto& i_data = I[io];
    rowgrp_nnz_rows.resize(nnz_row_sizes_loc[io]);
    Integer_Type k = 0;
    for(Integer_Type i = 0; i < tile_height; i++) {
        if(i_data[i]) {
            rowgrp_nnz_rows[k] = i;
            k++;
        }
    }    
    
    if(Env::is_master)
        printf("Vertex filtering: Filtering nonzero columns\n");
    J.resize(tiling->rank_ncolgrps);
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        J[i].resize(tile_width);
    JV.resize(tiling->rank_ncolgrps);
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        JV[i].resize(tile_width);    
    filter_vertices(_COLS_);
    uint32_t jo = accu_segment_col;
    auto& j_data = J[jo];
    colgrp_nnz_columns.resize(nnz_col_sizes_loc[jo]);
    k = 0;
    for(Integer_Type j = 0; j < tile_width; j++) {
        if(j_data[j]) {
            colgrp_nnz_columns[k] = j;
            k++;
        }
    }
        
    classify_vertices();

    rowgrp_regular_rows = regular_rows[io];
    rowgrp_source_rows = source_rows[io];
    colgrp_sink_columns = sink_columns[jo];    
}   


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter_rows() {
    MPI_Datatype TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    MPI_Datatype TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;

    std::vector<char> F(tile_height);
    std::vector<std::vector<char>> F_all;
    if(!Env::rank) {
        F_all.resize(tiling->rowgrp_nranks, std::vector<char>(tile_height));
        //for(uint32_t i = 0; i < tiling->rowgrp_nranks; i++) {
        //    F_all[i].resize(tile_length, 0);
        //}
    }
   
    auto &tile = tiles[0][Env::rank];
    for (auto& triple : *(tile.triples)) {
        if(!F[triple.row]) {
            F[triple.row] = 1;
        }
    }
    /*
    if(Env::rank == 2) {
        for(int i = 0; i < tiling->rowgrp_nranks - 1; i++) {
            printf("%d ", follower_rowgrp_ranks[i]);
        }
        printf("\n");
    }
    */
    int32_t leader, follower;//, my_rank, accu, this_segment;
    leader = 0;
    if(Env::is_master) {
        for(int32_t j = 0; j < Env::nranks - 1; j++) {
            follower = follower_rowgrp_ranks[j];
            auto& fj_data = F_all[j];
            Integer_Type fj_nitems = F_all[j].size();
            MPI_Irecv(fj_data.data(), fj_nitems, TYPE_CHAR, follower, 0, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    else {
        auto& f_data = F;
        Integer_Type f_nitems = F.size();
        MPI_Isend(f_data.data(), f_nitems, TYPE_CHAR, leader, 0, Env::MPI_WORLD, &request);
        out_requests.push_back(request);
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    
    
    
    if(Env::is_master) {
        
        for(int32_t j = 0; j < Env::nranks - 1; j++) {
            //accu = follower_rowgrp_ranks[j];
            auto &fj_data = F_all[j];
            Integer_Type fj_nitems = F_all[j].size();                  
            for(uint32_t i = 0; i < fj_nitems; i++) {
                if(fj_data[i] and !F[i])
                    F[i] = 1;
            }
        }
    }
    
    
    if(Env::is_master) {
        for(int32_t j = 0; j < Env::nranks - 1; j++) {
            follower = follower_rowgrp_ranks[j];
            auto& f_data = F;
            Integer_Type f_nitems = F.size();
            MPI_Isend(f_data.data(), f_nitems, TYPE_CHAR, follower, 0, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }
    }
    else {
        std::fill(F.begin(), F.end(), 0);
        auto& f_data = F;
        Integer_Type f_nitems = F.size();
        MPI_Irecv(f_data.data(), f_nitems, TYPE_CHAR, leader, 0, Env::MPI_WORLD, &request);
        out_requests.push_back(request);
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();

    rows_sizes.resize(Env::nranks);
    rows_sizes[Env::rank] = tile_width;
    for (int32_t j = 0; j < Env::nranks; j++) {
        if (j != Env::rank)
            MPI_Sendrecv(&rows_sizes[Env::rank], 1, TYPE_INT, j, 0, &rows_sizes[j], 1, TYPE_INT, 
                                                                         j, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
    }
    Env::barrier();
    
    
    
    start_dense.resize(Env::nranks);
    end_dense.resize(Env::nranks);
    for(int32_t i = 0; i < Env::nranks; i++) {
        if(i == 0)
            start_dense[i] = 0;
        else 
            start_dense[i] = std::accumulate(rows_sizes.begin(), rows_sizes.end() - Env::nranks + i, 0);
        
        end_dense[i] = std::accumulate(rows_sizes.begin(), rows_sizes.end() - Env::nranks + i + 1, 0);
    }
    
    /*
    Integer_Type start = 0;
    Integer_Type end = 0;
    if(!Env::rank) {
        start = 0;
    }
    else {
        start = std::accumulate(rows_sizes.begin(), rows_sizes.end() - Env::nranks + Env::rank, 0);
    }
    end = std::accumulate(rows_sizes.begin(), rows_sizes.end() - Env::nranks + Env::rank + 1, 0);
    
    printf("%d start=%d end=%d\n", Env::rank, start, end);
    */
    
    nnz_rows_sizes.resize(Env::nranks, 0);
    II.resize(tile_height , 0);
    II = F;
    F.clear();
    F.shrink_to_fit();
    IIV.resize(tile_height, 0);
    Integer_Type k = 0;
    for(Integer_Type i = 0; i < tile_height; i++) {
        if(II[i]) {
            IIV[i] = k; 
            k++;
            //rowgrp_nnz_rows.push_back(i);
        }
        if((i >= start_dense[Env::rank]) and (i < end_dense[Env::rank]) and II[i]) {
            nnz_rows_sizes[Env::rank]++;
        }
        
    }
    nnz_rows_size = k;
    
    for (int32_t j = 0; j < Env::nranks; j++) {
        if (j != Env::rank)
            MPI_Sendrecv(&nnz_rows_sizes[Env::rank], 1, TYPE_INT, j, 0, &nnz_rows_sizes[j], 1, TYPE_INT, 
                                                                         j, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
    }
    Env::barrier();
    
    nnz_rows_values.resize(nnz_rows_size);
    rowgrp_nnz_rows.resize(nnz_rows_sizes[Env::rank]);
    k = 0;
    Integer_Type l = 0;
    for(Integer_Type i = 0; i < tile_height; i++) {
        if(II[i]) {
            nnz_rows_values[l] = i;
            l++;
        }
        if((i >= start_dense[Env::rank]) and (i < end_dense[Env::rank]) and II[i]) {
            rowgrp_nnz_rows[k] = i;
            k++;
        }
    }
    
    k = 0;
    l = 0;
    start_sparse.resize(Env::nranks);
    end_sparse.resize(Env::nranks);
    for(int32_t i = 0; i < Env::nranks; i++) {
        if(i == 0)
            start_sparse[i] = 0;//nnz_rows_values[0];
        else 
            start_sparse[i] = std::accumulate(nnz_rows_sizes.begin(), nnz_rows_sizes.end() - Env::nranks + i, 0);
        
        end_sparse[i] = std::accumulate(nnz_rows_sizes.begin(), nnz_rows_sizes.end() - Env::nranks + i + 1, 0);
    }
    
    
    
    IT.resize(Env::nthreads);
    IVT.resize(Env::nthreads);
    threads_nnz_rows.resize(Env::nthreads);
    //threads_nnz_rows
    //threads_start_row
    //threads_end_row
    //if(!Env::rank) {
    std::vector<int> all_rows(Env::nthreads);
    
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        Integer_Type length = threads_end_row[tid] - threads_start_row[tid];
        /*
        if(tid == 0) {
            length = threads_end_row[tid] + 1;
        }
        else {
            length = threads_end_row[tid] - threads_end_row[tid - 1];
        }
        */
        //if(!Env::rank)
        //printf("rank=%d tid=%d start=%d end=%d sz=%d len=%d\n", Env::rank, tid, threads_start_row[tid], threads_end_row[tid], threads_end_row[tid] - threads_start_row[tid], length);
        IT[tid].resize(length);
        IVT[tid].resize(length);            

        Integer_Type j = 0;
        Integer_Type k = 0;
        for(Integer_Type i = threads_start_row[tid]; i < threads_end_row[tid]; i++) {
            if(II[i]) {
                IT[tid][j] = 1;
                IVT[tid][j] = k;
                k++;
            }
            j++;
        }
        threads_nnz_rows[tid] = k;
        all_rows[tid] = length;
        //if(tid == 10) {
        //    for(int j = 0; j < IT[tid].size(); j++)
        //        printf("%d %d %d\n", j, IT[tid][j], IVT[tid][j]);
        //}
    }
    
    int s1 = 0;
    for(int i =0; i < Env::nthreads; i++)
        s1 += threads_end_row[i] - threads_start_row[i];
    
    int s2 = 0;
    for(int i =0; i < Env::nthreads; i++)
        s2 += all_rows[i];
    
    int s = std::accumulate(threads_nnz_rows.begin(), threads_nnz_rows.end(), 0);
    //printf("%d %d %d %d\n", s, nnz_rows_size, s1, s2);
    
    threads_nnz_start_row.resize(Env::nthreads, 0);
    //if(!Env::rank) {
    Integer_Type nzz_sum = 0;
    for(int32_t i = 0; i < Env::nthreads; i++) {
        
        threads_nnz_start_row[i] += nzz_sum;
        nzz_sum += threads_nnz_rows[i];
        //printf("%d %d %d\n", i, threads_nnz_start_row[i], threads_nnz_rows[i]);
    }
    //}
    
    
    
    //Env::barrier();
    //Env::exit(0);
        
   // }
    
    //
 
    
        
    
    
    

    
    
    
    
    //printf("rank=%d rows_sizes=%d nnz_rows_sizes=%d\n", Env::rank, rows_sizes[Env::rank], nnz_rows_sizes[Env::rank]);
    //printf("rank=%d start=%d end=%d start=%d end=%d\n", Env::rank, start_dense[Env::rank], end_dense[Env::rank], start_sparse[Env::rank], end_sparse[Env::rank]);
    
    
    //nnz_rows_sizes[Env::rank] = k;

    
    
    //if(Env::is_master) {
      //  for(int i = 0; i < 
    //}
    
    //for (int32_t j = 0; j < Env::nranks - 1; j++) {
    
    
    
    /*
    if(!Env::rank) {
        for(int32_t j = 0; j < Env::nranks; j++) {
            printf("%d ", nnz_rows_sizes[j]);
        }
        printf("\n");
    }
    
    */
    

    
    /*
    if(!Env::rank) {
        for(int32_t j = 0; j < Env::nranks; j++) {
            printf("%d ", rows_sizes[j]);
        }
        printf(" %d %d\n", std::accumulate(rows_sizes.begin(), rows_sizes.end(), 0), tile_height);
    }    
    */
    //Integer_Type rows_all_sum = std::accumulate(rows_sizes.begin(), rows_sizes.end(), 0);
    //assert(rows_all_sum == tile_height);
    //printf("%d rows_sizes=%d nnz_rows_sizes=%d\n", Env::rank, rows_sizes[Env::rank], nnz_rows_sizes[Env::rank]);
    
    
    /*
    rowgrp_nnz_rows.resize(nnz_row_sizes_loc[io]);
    Integer_Type k = 0;
    for(Integer_Type i = 0; i < tile_height; i++) {
        if(i_data[i]) {
            rowgrp_nnz_rows[k] = i;
            k++;
        }
    } 
    */
    
    /*
    I.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        I[i].resize(tile_height);
    IV.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        IV[i].resize(tile_height);
    */
    
    
    
    /*
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        nnz_sizes_loc.push_back(nnz_sizes_all[this_segment]);
    }
    if(nnz_sizes_all[owned_segment]) {
        uint32_t ko = accu_segment_row_;
        auto &kj_data =  (*K)[ko];
        auto &kvj_data = (*KV)[ko];
        Integer_Type j = 0;
        for(uint32_t i = 0; i < f_nitems; i++) {
            if(f_data[i]) {
                kj_data[i] = 1;
                kvj_data[i] = j; 
                j++;
            }
            else {
                kj_data[i] = 0;
                kvj_data[i] = 0;
            }
        }
        assert(j == nnz_sizes_all[owned_segment]);
    }
    
    */
    
   // Env::barrier();     
    
    
    
    
    
    
    
    

}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter_cols() {
    MPI_Datatype TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    std::vector<char> F(tile_width);
    //std::vector<std::vector<char>> F_all;
    //if(!Env::rank) {
      //  F_all.resize(tiling->rowgrp_nranks, std::vector<char>(tile_height));
        //for(uint32_t i = 0; i < tiling->rowgrp_nranks; i++) {
        //    F_all[i].resize(tile_length, 0);
        //}
    //}
   
    auto &tile = tiles[0][Env::rank];
    for (auto& triple : *(tile.triples)) {
        auto pair = rebase(triple);
        if(!F[pair.col]) {
            F[pair.col] = 1;
        }
    }
    
    JJ.resize(tile_width , 0);
    JJ = F;
    F.clear();
    F.shrink_to_fit();
    JJV.resize(tile_width, 0);
    Integer_Type k = 0;
    for(Integer_Type j = 0; j < tile_width; j++) {
        if(JJ[j]) {
            JJV[j] = k; 
            k++;
        }
    }  
    nnz_cols_size_loc = k;
    colgrp_nnz_columns.resize(nnz_cols_size_loc);
    k = 0;
    for(Integer_Type j = 0; j < tile_width; j++) {
        if(JJ[j]) {
            //colgrp_nnz_columns[k] = start_dense[Env::rank] + j;
            colgrp_nnz_columns[k] = j;
            k++;
        }
    }
    //nnz_cols_size = k;
    nnz_cols_sizes.resize(Env::nranks);
    nnz_cols_sizes[Env::rank] = nnz_cols_size_loc;
    
    for (int32_t j = 0; j < Env::nranks; j++) {
        if (j != Env::rank)
            MPI_Sendrecv(&nnz_cols_sizes[Env::rank], 1, TYPE_INT, j, 0, &nnz_cols_sizes[j], 1, TYPE_INT, 
                                                                         j, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
    }
    Env::barrier();
    nnz_cols_size = std::accumulate(nnz_cols_sizes.begin(), nnz_cols_sizes.end(), 0);
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter_vertices(Filtering_type filtering_type_) {
    std::vector<std::vector<char>> *K;
    std::vector<std::vector<Integer_Type>> *KV;
    uint32_t rank_nrowgrps_, rank_ncolgrps_;
    uint32_t rowgrp_nranks_;
    Integer_Type tile_length;
    std::vector<int32_t> local_row_segments_;
    std::vector<int32_t> all_rowgrp_ranks_accu_seg_;
    std::vector<uint32_t> local_tiles_row_order_;
    int32_t accu_segment_rg_, accu_segment_row_;
    std::vector<int32_t> follower_rowgrp_ranks_; 
    std::vector<int32_t> follower_rowgrp_ranks_accu_seg_;
    std::vector<Integer_Type> nnz_sizes_all, nnz_sizes_loc;
    if(filtering_type_ == _ROWS_) {
        K = &I;
        KV = &IV;
        rank_nrowgrps_ = tiling->rank_nrowgrps;
        rank_ncolgrps_ = tiling->rank_ncolgrps;
        rowgrp_nranks_ = tiling->rowgrp_nranks;
        tile_length = tile_height;
        local_row_segments_ = local_row_segments;
        all_rowgrp_ranks_accu_seg_ = all_rowgrp_ranks_accu_seg;
        follower_rowgrp_ranks_ = follower_rowgrp_ranks;
        follower_rowgrp_ranks_accu_seg_ = follower_rowgrp_ranks_accu_seg;
        local_tiles_row_order_ = local_tiles_row_order;  
        accu_segment_rg_ = accu_segment_rg;
        accu_segment_row_ = accu_segment_row;
    }
    else if(filtering_type_ == _COLS_) {
        K = &J;
        KV = &JV;
        rank_nrowgrps_ = tiling->rank_ncolgrps;
        rank_ncolgrps_ = tiling->rank_nrowgrps;
        rowgrp_nranks_ = tiling->colgrp_nranks;
        tile_length = tile_width;
        local_row_segments_ = local_col_segments;
        all_rowgrp_ranks_accu_seg_ = all_colgrp_ranks_accu_seg;
        follower_rowgrp_ranks_ = follower_colgrp_ranks;
        follower_rowgrp_ranks_accu_seg_ = follower_colgrp_ranks_accu_seg;
        local_tiles_row_order_ = local_tiles_col_order;  
        accu_segment_rg_ = accu_segment_cg;
        accu_segment_row_ = accu_segment_col;
    }
    
    MPI_Datatype TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    MPI_Datatype TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;
    int32_t leader, follower, my_rank, accu, this_segment;
    uint32_t tile_th, pair_idx;
    bool vec_owner, communication;
    uint32_t fi = 0, fo = 0;

    
    /* F is a temprorary 3D array designated to the filtering step. We'd 
       rather use char for this because of narrowing down the heap usage.*/
    std::vector<std::vector<std::vector<char>>> F;
    F.resize(rank_nrowgrps_);
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        if(local_row_segments_[j] == owned_segment) {
            F[j].resize(rowgrp_nranks_);
            for(uint32_t i = 0; i < rowgrp_nranks_; i++)
                F[j][i].resize(tile_length, 0);
        }
        else {
            F[j].resize(1);
            F[j][0].resize(tile_length, 0);
        }
    }
    

    for(uint32_t t: local_tiles_row_order_) {
        auto pair = tile_of_local_tile(t);
        auto &tile = tiles[pair.row][pair.col];
        if(filtering_type_ == _ROWS_) {
            tile_th = tile.nth;
            pair_idx = pair.row;
        }
        else if(filtering_type_ == _COLS_) {
            tile_th = tile.mth;
            pair_idx = pair.col;
        }
        vec_owner = (leader_ranks[pair_idx] == Env::rank);
        if(vec_owner)
            fo = accu_segment_rg_;
        else
            fo = 0;
        auto &f_data = F[fi][fo];
        Integer_Type f_nitems = tile_length;
        if(filtering_type_ == _ROWS_) {
            for (auto& triple : *(tile.triples)) {
                test(triple);
                auto pair1 = rebase(triple);
                if(!f_data[pair1.row])
                    f_data[pair1.row] = 1;
            }
        }
        else if(filtering_type_ == _COLS_) {
            for (auto& triple : *(tile.triples)) {
                test(triple);
                auto pair1 = rebase(triple);
                if(!f_data[pair1.col]) {
                    f_data[pair1.col] = 1;
                }
            }
        }        
        communication = (((tile_th + 1) % rank_ncolgrps_) == 0);
        if(communication) {
            if(filtering_type_ == _ROWS_)
                leader = tile.leader_rank_rg;
            if(filtering_type_ == _COLS_)
                leader = tile.leader_rank_cg;
            my_rank = Env::rank;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++) {
                    follower = follower_rowgrp_ranks_[j];
                    accu = follower_rowgrp_ranks_accu_seg_[j];
                    auto &fj_data = F[fi][accu];
                    Integer_Type fj_nitems = tile_length;
                    MPI_Irecv(fj_data.data(), fj_nitems, TYPE_CHAR, follower, pair_idx, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }   
            else {
                MPI_Isend(f_data.data(), f_nitems, TYPE_CHAR, leader, pair_idx, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            fi++;
        }
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
    fi = accu_segment_row_;
    fo = accu_segment_rg_;
    auto &f_data = F[fi][fo];
    Integer_Type f_nitems = tile_length;
    for(uint32_t j = 0; j < rowgrp_nranks_ - 1; j++) {
        accu = follower_rowgrp_ranks_accu_seg_[j];
        auto &fj_data = F[fi][accu];
        Integer_Type fj_nitems = tile_length;                  
        for(uint32_t i = 0; i < fj_nitems; i++) {
            if(fj_data[i] and !f_data[i])
                f_data[i] = 1;
        }
    }    
    Integer_Type nnz_local = 0;
    for(uint32_t i = 0; i < f_nitems; i++) {
        if(f_data[i])
            nnz_local++;
    }
    nnz_sizes_all.resize(Env::nranks);
    nnz_sizes_all[owned_segment] = nnz_local;
    Env::barrier();     
    for (int32_t j = 0; j < Env::nranks; j++) {
        int32_t r = leader_ranks[j];
        if (j != owned_segment)
            MPI_Sendrecv(&nnz_sizes_all[owned_segment], 1, TYPE_INT, r, Env::rank, &nnz_sizes_all[j], 1, TYPE_INT, 
                                                                         r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
    }
    Env::barrier();     
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        nnz_sizes_loc.push_back(nnz_sizes_all[this_segment]);
    }
    if(nnz_sizes_all[owned_segment]) {
        uint32_t ko = accu_segment_row_;
        auto &kj_data =  (*K)[ko];
        auto &kvj_data = (*KV)[ko];
        Integer_Type j = 0;
        for(uint32_t i = 0; i < f_nitems; i++) {
            if(f_data[i]) {
                kj_data[i] = 1;
                kvj_data[i] = j; 
                j++;
            }
            else {
                kj_data[i] = 0;
                kvj_data[i] = 0;
            }
        }
        assert(j == nnz_sizes_all[owned_segment]);
    }
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        auto &kj_data = (*K)[j];
        if(this_segment == owned_segment) {
            for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++) {
                follower = follower_rowgrp_ranks_[i];
                MPI_Isend(kj_data.data(), tile_length, TYPE_CHAR, follower, owned_segment, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Irecv(kj_data.data(), tile_length, TYPE_CHAR, leader, this_segment, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    } 
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        auto &kvj_data = (*KV)[j];
        if(this_segment == owned_segment) {
            for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++) {
                follower = follower_rowgrp_ranks_[i];
                MPI_Isend(kvj_data.data(), tile_length, TYPE_INT, follower, owned_segment, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Irecv(kvj_data.data(), tile_length, TYPE_INT, leader, this_segment, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    } 
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear();
    for (uint32_t j = 0; j < rank_nrowgrps_; j++)
    {
        auto &kj_data = (*K)[j];
        auto &kvj_data = (*KV)[j];
        Integer_Type k = 0;
        for(uint32_t i = 0; i < tile_length; i++) {
            if(kj_data[i]) {
                assert(kvj_data[i] == k);
                k++;
            }
            else
                assert(kvj_data[i] == 0);
        }
    }
    if(filtering_type_ == _ROWS_) {
        nnz_row_sizes_all = nnz_sizes_all;
        nnz_row_sizes_loc = nnz_sizes_loc;
    }
    else if(filtering_type_ == _COLS_) {
        nnz_col_sizes_all = nnz_sizes_all;
        nnz_col_sizes_loc = nnz_sizes_loc;
    }
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        if(local_row_segments_[j] == owned_segment) {
            for(uint32_t i = 0; i < rowgrp_nranks_; i++) {
                F[j][i].clear();
                F[j][i].shrink_to_fit();
            }
        }
        else {
            F[j][0].clear();
            F[j][0].shrink_to_fit();
        }
    }
    F.clear();
    F.shrink_to_fit();
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::classify_vertices() {
    uint32_t io = accu_segment_row;
    auto& i_data = I[io];
    uint32_t jo = accu_segment_col;
    auto& j_data = J[jo];
    
    regular_rows.resize(tiling->rank_nrowgrps);
    source_rows.resize(tiling->rank_nrowgrps);
    regular_columns.resize(tiling->rank_ncolgrps);
    sink_columns.resize(tiling->rank_ncolgrps);
    for(Integer_Type i = 0; i < tile_height; i++) {
        if(i_data[i] and j_data[i]) {
            regular_rows[io].push_back(i);
            regular_columns[jo].push_back(i);
        }
        if(i_data[i] and !j_data[i])
            source_rows[io].push_back(i);
        if(!i_data[i] and j_data[i])
            sink_columns[jo].push_back(i);
    }
    // regular rows/cols
    MPI_Datatype TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    MPI_Request request;
    MPI_Status status;
    int32_t leader, my_rank, row_group, col_group;
    int32_t follower;
    Integer_Type nitems = 0;
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        row_group = local_row_segments[i];
        leader = leader_ranks[row_group];
        my_rank = Env::rank;
        if(leader == my_rank) {            
            for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++) {
                follower = follower_rowgrp_ranks[j];
                nitems = regular_rows[io].size();
                MPI_Send(&nitems, 1, TYPE_INT, follower, row_group, Env::MPI_WORLD);
                MPI_Isend(regular_rows[io].data(), regular_rows[io].size(), TYPE_INT, follower, row_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, row_group, Env::MPI_WORLD, &status);
            regular_rows[i].resize(nitems);
            MPI_Irecv(regular_rows[i].data(), regular_rows[i].size(), TYPE_INT, leader, row_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
        
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    regular_rows_bitvector.resize(tiling->rank_nrowgrps);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        regular_rows_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        for(uint32_t j: regular_rows[i])
            regular_rows_bitvector[i][j] = 1;
    }
    // Source rows
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        row_group = local_row_segments[i];
        leader = leader_ranks[row_group];
        my_rank = Env::rank;
        if(leader == my_rank) {            
            for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++) {
                follower = follower_rowgrp_ranks[j];
                nitems = source_rows[io].size();
                MPI_Send(&nitems, 1, TYPE_INT, follower, row_group, Env::MPI_WORLD);
                MPI_Isend(source_rows[io].data(), source_rows[io].size(), TYPE_INT, follower, row_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, row_group, Env::MPI_WORLD, &status);
            source_rows[i].resize(nitems);
            MPI_Irecv(source_rows[i].data(), source_rows[i].size(), TYPE_INT, leader, row_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    source_rows_bitvector.resize(tiling->rank_nrowgrps);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        source_rows_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        for(Integer_Type j: source_rows[i])
                source_rows_bitvector[i][j] = 1;
    }
    // Regular columns
    for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++) {
        col_group = local_col_segments[jo];
        follower = follower_colgrp_ranks[i];
        nitems = regular_columns[jo].size();
        MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
        MPI_Isend(regular_columns[jo].data(), regular_columns[jo].size(), TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
        out_requests.push_back(request);
    }
    
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks[col_group];
        my_rank = Env::rank;
        if(leader != my_rank) {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
            regular_columns[i].resize(nitems);
            MPI_Irecv(regular_columns[i].data(), regular_columns[i].size(), TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    regular_columns_bitvector.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        regular_columns_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        for(Integer_Type j: regular_columns[i])
            regular_columns_bitvector[i][j] = 1;
    }     
    // Sink columns
    for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++) {
        col_group = local_col_segments[jo];
        follower = follower_colgrp_ranks[i];
        nitems = sink_columns[jo].size();
        MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
        MPI_Isend(sink_columns[jo].data(), nitems, TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
        out_requests.push_back(request);
    }
    
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks[col_group];
        my_rank = Env::rank;
        if(leader != my_rank) {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
            sink_columns[i].resize(nitems);
            MPI_Irecv(sink_columns[i].data(), sink_columns[i].size(), TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    sink_columns_bitvector.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        sink_columns_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        for(Integer_Type j: sink_columns[i])
            sink_columns_bitvector[i][j] = 1;
    }    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression() {
    if(compression_type == _CSC_) {
        if(Env::is_master)
            printf("Edge compression: CSC\n");     
        init_csc();
    } 
    else if(compression_type == _DCSC_) {
        if(Env::is_master)
            printf("Edge compression: DCSC\n");
        init_dcsc();
    }   
    else if(compression_type == _TCSC_){
        if(Env::is_master)
            printf("Edge compression: TCSC\n");
        init_tcsc();
    }
    else {
        if(Env::is_master)
            printf("Edge compression: TCSC_CF\n");
        init_tcsc_cf();
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_csc() {
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.compressor = new CSC_BASE<Weight, Integer_Type>(tile.nedges, tile_width);
        if(tile.nedges)
            tile.compressor->populate(tile.triples, tile_height, tile_width);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_dcsc() {
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        Integer_Type c_nitems = nnz_col_sizes_loc[xi];
        auto& j_data = J[xi];
        auto& jv_data = JV[xi];
        tile.compressor = new DCSC_BASE<Weight, Integer_Type>(tile.nedges, c_nitems);
        if(tile.nedges)
            tile.compressor->populate(tile.triples, tile_height, tile_width, j_data, jv_data);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row)
            xi = 0;
    }   
}



template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc()
{
    //printf("[+]init_tcsc\n");
    /*
    auto& tile = tiles[0][Env::rank];
    std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
    if(triples.size()) {
        tile.compressor_t.resize(tile.npartitions);
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            int cid = sched_getcpu();
            int sid =  cid / Env::nthreads_per_socket;
            
            Integer_Type c_nitems = nnz_cols_sizes[Env::rank];
            //Integer_Type r_nitems = nnz_rows_size;
            Integer_Type r_nitems = threads_nnz_rows[tid];
            Integer_Type tile_height_t = (tid == 0) ? 0 : threads_end_row[tid - 1];
            struct Triple<Weight, Integer_Type> f = tile.triples_t[tid]->front();
            //struct Triple<Weight, Integer_Type> b = tile.triples_t[tid]->back();
           // printf("%d %d %d %d %d\n", tid, tile_height_t, tile.triples_t[tid]->size(), f.row, f.col);
            auto& i_data = IT[tid];
            auto& iv_data = IVT[tid];
            auto& j_data = JJ;
            auto& jv_data = JJV;
            
            
            
            tile.compressor_t[tid] = new TCSC_BASE<Weight, Integer_Type>(tile.triples_t[tid]->size(), c_nitems, r_nitems, sid);
            tile.compressor_t[tid]->populate(tile.triples_t[tid], tile_height_t, tile_width, i_data, iv_data, j_data, jv_data);
            
            Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
            Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
            Integer_Type nnzcols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;
        }
    }
    */
    

    //Env::barrier();
    //Env::exit(0);
    /*
    if(tid == 0) {
            length = threads_end_row[tid] + 1;
        }
        else {
            length = threads_end_row[tid] - threads_end_row[tid - 1];
        }
    */
    
    /*
    if(!Env::rank){
        int tid = 11;
        Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
        Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
        Integer_Type nnzcols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;
        
        for(uint32_t j = 0; j < nnzcols; j++) {
            printf("j=%d\n", j);
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                printf(" IA[%d]=%d, j=%d\n", i, IA[i], j);
            }
        }
    }
    */
    
    
    
        
    //Env::barrier();
    //Env::exit(0);    
    
   
    
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        Integer_Type c_nitems = nnz_col_sizes_loc[xi];
        Integer_Type r_nitems = nnz_row_sizes_loc[yi];
        auto& i_data = I[yi];
        auto& iv_data = IV[yi];
        auto& j_data = J[xi];
        auto& jv_data = JV[xi];
        if(tile.nedges) {
            tile.compressor_t.resize(tile.npartitions);
            #pragma omp parallel 
            {
                int tid = omp_get_thread_num();
                int cid = sched_getcpu();
                int sid =  cid / Env::nthreads_per_socket;
                
                /*
                if(sid)
                    sid = 0;
                else
                    sid = 1;
                */
                //printf("%d nnz=%d\n", tid, tile.triples_t[tid]->size());
                
                tile.compressor_t[tid] = new TCSC_BASE<Weight, Integer_Type>(tile.triples_t[tid]->size(), c_nitems, r_nitems, sid);
                tile.compressor_t[tid]->populate(tile.triples_t[tid], tile_height, tile_width, i_data, iv_data, j_data, jv_data);
                //numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
                //numa_tonode_memory
                
                
                Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                Integer_Type nnzcols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;
                
                //if(tid == 3) {
                    /*
                    int r = 0;
                    int c = 0;
                for(uint32_t j = 0; j < nnzcols; j++) {
                    //printf("j=%d\n", j);
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        //printf(" %d %d\n", JA[j], IA[i]);
                        if(IA[i]> r);
                        r = IA[i];
                    }
                }
                */
                //printf("%d %d %d\n", tid, r, r_nitems);
                //}
                
                
                
            }
        }
        
        /*
        #pragma omp parallel 
        {
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            
            uint32_t rows_per_thread = tile_height/nthreads;
            
            uint32_t start = tid * rows_per_thread;
            uint32_t end = start + rows_per_thread;
            start = (start > tile_height) ? (tile_height):(start);
            end = (end > tile_height) ? (tile_height):(end);
            end = (tid == nthreads - 1)?(tile_height):(end);
            printf("thread %d of %d in [%d %d]\n", tid, nthreads, start, end);
        }
        */
        
        
        /*
        tile.compressor = new TCSC_BASE<Weight, Integer_Type>(tile.nedges, c_nitems, r_nitems);
        
        if(tile.nedges)
            tile.compressor->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data);
        */
        
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
            yi++;
        }
        //printf("np= %d\n", tile.npartitions);
    }  
    //tilet.riples_t[tid]
    del_triples_t();
    del_classifier();
    
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf()
{
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order)
    {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        Integer_Type c_nitems = nnz_col_sizes_loc[xi];
        Integer_Type r_nitems = nnz_row_sizes_loc[yi];
        auto& i_data = I[yi];
        auto& iv_data = IV[yi];
        auto& j_data = J[xi];
        auto& jv_data = JV[xi];
        auto& regular_rows_data = regular_rows[yi];
        auto& regular_rows_bv_data = regular_rows_bitvector[yi];
        auto& source_rows_data = source_rows[yi];
        auto& source_rows_bv_data = source_rows_bitvector[yi];
        auto& regular_columns_data = regular_columns[xi];
        auto& regular_columns_bv_data = regular_columns_bitvector[xi];
        auto& sink_columns_data = sink_columns[xi];
        auto& sink_columns_bv_data = sink_columns_bitvector[xi];
        tile.compressor = new TCSC_CF_BASE<Weight, Integer_Type>(tile.nedges, c_nitems, r_nitems);
        if(tile.nedges)
            tile.compressor->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data, regular_rows_data, regular_rows_bv_data, source_rows_data, source_rows_bv_data, regular_columns_data, regular_columns_bv_data, sink_columns_data, sink_columns_bv_data);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
            yi++;
        }
    }  
    del_classifier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_classifier() {
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        regular_rows[i].clear();
        regular_rows[i].shrink_to_fit();
        regular_rows_bitvector[i].clear();
        regular_rows_bitvector[i].shrink_to_fit();
    }
    regular_rows.clear();
    regular_rows.shrink_to_fit();    
    regular_rows_bitvector.clear();
    regular_rows_bitvector.shrink_to_fit();    
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        source_rows[i].clear();
        source_rows[i].shrink_to_fit();
        source_rows_bitvector[i].clear();
        source_rows_bitvector[i].shrink_to_fit();
    }
    source_rows.clear();
    source_rows.shrink_to_fit();
    source_rows_bitvector.clear();
    source_rows_bitvector.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        regular_columns[i].clear();
        regular_columns[i].shrink_to_fit();
        regular_columns_bitvector[i].clear();
        regular_columns_bitvector[i].shrink_to_fit();
    }
    regular_columns.clear();
    regular_columns.shrink_to_fit();
    regular_columns_bitvector.clear();
    regular_columns_bitvector.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        sink_columns[i].clear();
        sink_columns[i].shrink_to_fit();
        sink_columns_bitvector[i].clear();
        sink_columns_bitvector[i].shrink_to_fit();
    }
    sink_columns.clear();
    sink_columns.shrink_to_fit();
    sink_columns_bitvector.clear();
    sink_columns_bitvector.shrink_to_fit();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_filter() {
    
    /*
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        I[i].clear();
        I[i].shrink_to_fit();
    }
    I.clear();
    I.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        J[i].clear();
        J[i].shrink_to_fit();
    }
    J.clear();
    J.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {   
        IV[i].clear();
        IV[i].shrink_to_fit();
    }
    IV.clear();
    IV.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {   
        JV[i].clear();
        JV[i].shrink_to_fit();
    }
    JV.clear();
    JV.shrink_to_fit();
    
    rowgrp_nnz_rows.clear();
    rowgrp_nnz_rows.shrink_to_fit();
    rowgrp_regular_rows.clear();
    rowgrp_regular_rows.shrink_to_fit();
    rowgrp_source_rows.clear();
    rowgrp_source_rows.shrink_to_fit();
    colgrp_nnz_columns.clear();
    colgrp_nnz_columns.shrink_to_fit();
    colgrp_sink_columns.clear();
    colgrp_sink_columns.shrink_to_fit();
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_compression() {
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        delete tile.compressor;
        tile.compressor = nullptr;
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_triples() {
    
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        tile.free_triples();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_triples_t() {
    /*
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.nedges) {
            #pragma omp parallel 
            {
                int tid = omp_get_thread_num();
                tile.triples_t[tid]->clear();
                tile.triples_t[tid]->shrink_to_fit();
                delete tile.triples_t[tid];
                tile.triples_t[tid] = nullptr;
            }
        }
    }
    */
}


#endif
