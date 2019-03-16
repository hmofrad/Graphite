/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include "mpi/types.hpp" 
#include "mat/tiling.hpp" 
#include "ds/indexed_sort.hpp"
#include <ds/vector.hpp>
#include <ds/segment.hpp>
enum Filtering_type
{
  _ROWS_,
  _COLS_
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
    int32_t thread;
    int32_t leader;
    int32_t leader_rank_rg, leader_rank_cg;
    int32_t rank_rg, rank_cg;
    int32_t leader_rank_rg_rg, leader_rank_cg_cg;
    uint64_t nedges;
    void allocate_triples();
    void free_triples();
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
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__, typename Vertex_State, typename Vertex_Methods_Impl>
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
        
        std::vector<Integer_Type> rows_sizes;
        Integer_Type rows_size;
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_rows_values;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        Integer_Type nnz_cols_size_loc;

        
        
        std::vector<Segment<Weight, Integer_Type, char>*> I1;
        std::vector<Segment<Weight, Integer_Type, Integer_Type>*> IV1;
        std::vector<Segment<Weight, Integer_Type, char>*> J1;
        std::vector<Segment<Weight, Integer_Type, Integer_Type>*> JV1;
        std::vector<Segment<Weight, Integer_Type, Integer_Type>*> rowgrp_nnz_rows1;
        std::vector<Segment<Weight, Integer_Type, Integer_Type>*> colgrp_nnz_cols1;
        
        
        Vector<Weight, Integer_Type, char>* I = nullptr;
        Vector<Weight, Integer_Type, Integer_Type>* IV = nullptr;
        Vector<Weight, Integer_Type, char>* J = nullptr;
        Vector<Weight, Integer_Type, Integer_Type>* JV = nullptr;
        Vector<Weight, Integer_Type, Integer_Type>* rowgrp_nnz_rows;
        Vector<Weight, Integer_Type, Integer_Type>* colgrp_nnz_cols;
        
        struct Segment<Weight, Integer_Type, char>** I2;
        struct Segment<Weight, Integer_Type, Integer_Type>** IV2;
        struct Segment<Weight, Integer_Type, char>** J2;
        struct Segment<Weight, Integer_Type, Integer_Type>** JV2;
        struct Segment<Weight, Integer_Type, Integer_Type>** rowgrp_nnz_rows2;
        struct Segment<Weight, Integer_Type, Integer_Type>** colgrp_nnz_cols2;
        
        /*
        std::vector<std::vector<char>> I;           // Nonzero rows bitvectors (from tile width)
        std::vector<std::vector<Integer_Type>> IV;  // Nonzero rows indices    (from tile width)
        std::vector<std::vector<char>> J;           // Nonzero cols bitvectors (from tile height)
        std::vector<std::vector<Integer_Type>> JV;  // Nonzero cols indices    (from tile height)
        */
        /*
        std::vector<Integer_Type> rowgrp_nnz_rows;  // Row group row indices         
        std::vector<Integer_Type> rowgrp_regular_rows; // Row group regular indices
        std::vector<Integer_Type> rowgrp_source_rows;  // Row group source column indices
        std::vector<Integer_Type> colgrp_nnz_columns;  // Column group column indices
        std::vector<Integer_Type> colgrp_sink_columns;    // Column group sink column indices
        
        
        std::vector<std::vector<Integer_Type>> rowgrp_nnz_rows_t;
        std::vector<std::vector<Integer_Type>> colgrp_nnz_cols_t;
        */
        std::vector<std::vector<struct Tile2D<Weight, Integer_Type, Fractional_Type>>> tiles;
        
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;
        std::vector<int32_t> local_row_segments;
        std::vector<int32_t> local_col_segments;
        std::vector<std::vector<uint32_t>> local_tiles_row_order_t;
        std::vector<std::vector<uint32_t>> local_tiles_col_order_t;
        
        std::vector<int32_t> leader_ranks;
        std::vector<int32_t> leader_ranks_row;
        std::vector<int32_t> leader_ranks_col;
        
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
        std::vector<int32_t> owned_segments, accu_segments, accu_segment_rows, accu_segment_cols;
        std::vector<int32_t> owned_segments_row, owned_segments_col;
        int32_t num_owned_segments;
        std::vector<int32_t> owned_segments_all;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        
        struct Triple<Weight, Integer_Type> base(const struct Triple<Weight, Integer_Type>& pair, Integer_Type rowgrp, Integer_Type colgrp);
        struct Triple<Weight, Integer_Type> rebase(const struct Triple<Weight, Integer_Type>& pair);
        void insert(const struct Triple<Weight, Integer_Type>& triple);
        void test(const struct Triple<Weight, Integer_Type>& triple);      
        struct Triple<Weight, Integer_Type> tile_of_local_tile(const uint32_t local_tile);
        struct Triple<Weight, Integer_Type> tile_of_triple(const struct Triple<Weight, Integer_Type>& triple);
        
        void free_tiling();
        void init_matrix();
        void del_triples();
        void init_tiles();
        void init_threads();
        void closest(std::vector<struct Triple<Weight, Integer_Type>>& triples, int32_t npartitions, uint64_t chunk_size_, 
                     std::vector<uint64_t>& start, std::vector<uint64_t>& end);
        void init_compression();
        void init_tcsc();
        void init_tcsc_threaded(int tid);
;
        void del_compression();
        void del_filter();
        void del_classifier();
        void print(std::string element);
        void distribute();
        void init_filtering();
        void filter_vertices(Filtering_type filtering_type_);
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, bool directed_, bool transpose_, bool parallel_edges_, Tiling_type tiling_type_, 
    Compression_type compression_type_) {
    nrows = nrows_;
    ncols = ncols_;
    ntiles = ntiles_;
    nrowgrps = sqrt(ntiles);
    ncolgrps = ntiles / nrowgrps;
    tile_height = nrows / nrowgrps;
    tile_width  = ncols / ncolgrps;
    directed = directed_;
    transpose = transpose_;
    parallel_edges = parallel_edges_;
    // Initialize tiling 
    tiling = new Tiling(Env::nranks, ntiles, nrowgrps, ncolgrps, tiling_type_);
    compression_type = compression_type_;
    // Initialize matrix
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
    if(tiles[pair.row][pair.col].rank != Env::rank) {
        printf("rank=%d: Invalid entry for tile[%d][%d]=[%d %d]\n", Env::rank, pair.row, pair.col, triple.row, triple.col);
        Env::exit(0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::insert(const struct Triple<Weight, Integer_Type>& triple) {
    struct Triple<Weight, Integer_Type> pair = tile_of_triple(triple);
    if(pair.row > tiling->nrowgrps or pair.col > tiling->ncolgrps) {
        printf("rank=%d: Invalid entry for tile[%d][%d]=[%d %d]\n", Env::rank, pair.row, pair.col, triple.row, triple.col);
        Env::exit(0);
    }
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
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks + (j % tiling->rowgrp_nranks);
            }
            else if(tiling->tiling_type == Tiling_type::_NUMA_) {
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks + (j % tiling->rowgrp_nranks);
                tile.rank = Env::ranks[tile.rank];
            }
            else {
                fprintf(stderr, "ERROR(rank=%d): Invalid tiling type\n", Env::rank);
                Env::exit(1);
            }
                            
            tile.thread = (i / tiling->colgrp_nranks) % Env::nthreads;
            
            tile.ith = tile.rg / tiling->colgrp_nranks; 
            tile.jth = tile.cg / tiling->rowgrp_nranks;
            
            tile.rank_rg = j % tiling->rowgrp_nranks;
            tile.rank_cg = i % tiling->colgrp_nranks;
            
            tile.leader_rank_rg = i;
            tile.leader_rank_cg = j;
            
            tile.leader_rank_rg_rg = i;
            tile.leader_rank_cg_cg = j;
            
            tile.nth = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.mth = (tile.jth * tiling->rank_nrowgrps) + tile.ith;
                        
            tile.allocate_triples();
        }
    }

    struct Triple<Weight, Integer_Type> pair;
    std::vector<int32_t> counts(Env::nranks);
    
    leader_ranks.resize(nrowgrps, -1);
    leader_ranks_rg.resize(nrowgrps);
    leader_ranks_cg.resize(ncolgrps);    
    leader_ranks_row.resize(nrowgrps);
    leader_ranks_col.resize(ncolgrps);

    /* Put unique ranks on the diagonal of 2D grid of tiles
       If there is more MPI ranks than the number of segments,
       use count vector to enforce the numbwe of owned segments*/
    num_owned_segments = nrowgrps / Env::nranks;
    assert(num_owned_segments == Env::nthreads);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = i; j < ncolgrps; j++) { 
            if(counts[tiles[j][i].rank] < num_owned_segments) {        
                counts[tiles[j][i].rank]++;
                std::swap(tiles[j], tiles[i]);
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
        leader_ranks_rg[i] = tiles[i][i].rank_rg;
        leader_ranks_cg[i] = tiles[i][i].rank_cg;
    }
    
    //Calculate local tiles in row order and local row and columns
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
                tile.rg = i;
                tile.cg = j;
                tile.kth   = (tile.rg * tiling->ncolgrps) + tile.cg;
                tile.leader = leader_ranks[i];
            if(tile.rank == Env::rank) {
                pair.row = i;
                pair.col = j;    
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
            if(i == j) {
                if(leader_ranks[i] == Env::rank) {
                    owned_segment = i;
                    owned_segments.push_back(owned_segment);
                }
            }
        }
    }
    assert(num_owned_segments == (int32_t) owned_segments.size());
    
    std::fill(counts.begin(), counts.end(), 0);
    owned_segments_all.resize(Env::nsegments);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            if(i == j) {
                auto& tile = tiles[i][j];
                int32_t r = tile.rank;
                int32_t k = counts[r] + (r * num_owned_segments);
                owned_segments_all[k] = i;
                counts[r]++;
            }
        }
    }
    

    //Calculate local tiles in column order
    for (uint32_t j = 0; j < ncolgrps; j++) {
        for (uint32_t i = 0; i < nrowgrps; i++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) {
                local_tiles_col_order.push_back(tile.kth);
            }
        }
    }
    
    // Calculate methadata required for processing tiles in vertex program
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

    /* Spilitting communicator among row/col groups and creating
       the methadata required for processing them in vertex program */
    /*
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
    */
    if(not Env::get_init_status()) {
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        Env::set_init_status();
    }
    // Which column index in my rowgrps is mine when I'm the accumulator
    for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++) {
        if(all_rowgrp_ranks[j] == Env::rank) {
            accu_segment_rg = j;
        }
        if(j+1 >= all_rowgrp_ranks.size())
            break;
    }
    // Which row index in my colgrps is mine when I'm the accumulator
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++) {
        if(all_colgrp_ranks[j] == Env::rank) {
            accu_segment_cg = j;
        }
        if(j+1 >= all_colgrp_ranks.size())
            break;
    } 
    // Which rowgrp is mine
    for(uint32_t j = 0; j < tiling->rank_nrowgrps; j++) {
        if(leader_ranks[local_row_segments[j]] == Env::rank) {
            accu_segment_row = j;
            accu_segment_rows.push_back(j);
        }
    }
    // Which colgrp is mine
    for(uint32_t j = 0; j < tiling->rank_ncolgrps; j++) {
        if(leader_ranks[local_col_segments[j]] == Env::rank) {
            accu_segment_col = j;
            accu_segment_cols.push_back(j);
        }
    } 
    
    // Distribute tiles among threads
    local_tiles_row_order_t.resize(Env::nthreads);    
    for(int32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        local_tiles_row_order_t[tile.mth % Env::nthreads].push_back(t);
    }
    local_tiles_col_order_t.resize(Env::nthreads);
    for(int32_t t: local_tiles_col_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        local_tiles_col_order_t[tile.nth % Env::nthreads].push_back(t);
    } 
  
    // Print tiling assignment
    if(Env::is_master) {
        printf("INFO(rank=%d): 2D tiling: %d x %d [nrows x ncols]\n", Env::rank, nrows, ncols);
        printf("INFO(rank=%d): 2D tiling: %d x %d [nrowgrps x ncolgrps]\n", Env::rank, nrowgrps, ncolgrps);
        printf("INFO(rank=%d): 2D tiling: %d x %d [nranks x nthreads] = %d = nsegments = nrowgrps = ncolgrps\n", Env::rank, Env::nranks, Env::nthreads, Env::nsegments);
        printf("INFO(rank=%d): 2D tiling: %d x %d [height x width]\n", Env::rank, tile_height, tile_width);
        printf("INFO(rank=%d): 2D tiling: %d x %d [rowgrp_nranks x colgrp_nranks]\n", Env::rank, tiling->rowgrp_nranks, tiling->colgrp_nranks);
        printf("INFO(rank=%d): 2D tiling: %d x %d [rank_nrowgrps x rank_ncolgrps]\n", Env::rank, tiling->rank_nrowgrps, tiling->rank_ncolgrps);
    }
    print("rank");
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::print(std::string element) {
    if(Env::is_master) {    
        uint32_t skip = 15;
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                if(element.compare("rank") == 0) 
                    printf("%02d ", tile.rank);
                if(element.compare("thread") == 0) 
                    printf("%02d ", tile.thread);
                else if(element.compare("kth") == 0) 
                    printf("%3d ", tile.kth);
                else if(element.compare("ith") == 0) 
                    printf("%2d ", tile.ith);
                else if(element.compare("jth") == 0) 
                    printf("%2d ", tile.jth);
                else if(element.compare("nth") == 0) 
                    printf("%2d ", tile.nth);
                else if(element.compare("mth") == 0) 
                    printf("%2d ", tile.mth);
                else if(element.compare("rank_rg") == 0) 
                    printf("%2d ", tile.rank_rg);
                else if(element.compare("rank_cg") == 0) 
                    printf("%2d ", tile.rank_cg);
                else if(element.compare("leader_rank_rg") == 0) 
                    printf("%2d ", tile.leader_rank_rg);
                else if(element.compare("leader_rank_cg") == 0) 
                    printf("%2d ", tile.leader_rank_cg);
                else if(element.compare("leader_rank_rg_rg") == 0) 
                    printf("%2d ", tile.leader_rank_rg_rg);
                else if(element.compare("leader_rank_cg_cg") == 0) 
                    printf("%2d ", tile.leader_rank_cg_cg);
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
    distribute();
    ColSort<Weight, Integer_Type> f_col;
    auto f_comp = [] (const Triple<Weight, Integer_Type> &a, const Triple<Weight, Integer_Type> &b)
                  {return (a.row == b.row and a.col == b.col);};
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
        if(triples.size()) {
            std::sort(triples.begin(), triples.end(), f_col);
            /* remove parallel edges (duplicates), necessary for triangle couting */
            if(not parallel_edges) {
                auto last = std::unique(triples.begin(), triples.end(), f_comp);
                triples.erase(last, triples.end());
            }
        }
        tile.nedges = tile.triples->size();
    }
}


/* Inspired by LA3 code @
   https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/dist_matrix2d.hpp
*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::distribute()
{
    MPI_Barrier(MPI_COMM_WORLD);
    if(Env::is_master)
        printf("INFO(rank=%d): Edge distribution: Distributing edges among %d ranks\n", Env::rank, Env::nranks);     
    
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

    // Populate tiles with received edges
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
    
    // Sanity check
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) {
                std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
                nedges_end_local += triples.size();
            }
        }
    }    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    MPI_Allreduce(&nedges_end_local, &nedges_end_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges_start_global == nedges_end_global);
    if(Env::is_master)
        printf("INFO(rank=%d): Edge distribution: Sanity check for exchanging %lu edges is done\n", Env::rank, nedges_end_global);
    auto retval = MPI_Type_free(&MANY_TRIPLES);
    assert(retval == MPI_SUCCESS);   
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_filtering() {
    if(Env::is_master)
        printf("INFO(rank=%d): Vertex filtering: Filtering zero rows/cols\n", Env::rank);
    
    /*
    std::vector<Integer_Type> i_sizes(tiling->rank_nrowgrps, tile_height);
    int num_rowgrps_per_thread = tiling->rank_nrowgrps / num_owned_segments;
    assert((num_rowgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_nrowgrps);
    std::vector<Integer_Type> all_rowgrps_thread_sockets(tiling->rank_nrowgrps);    
    for(int i = 0; i < num_rowgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_rowgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    
        
    I2 = (struct Segment<Weight, Integer_Type, char>**) mmap(nullptr, (tiling->rank_nrowgrps * sizeof(struct Segment<Weight, Integer_Type, char>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    IV2 = (struct Segment<Weight, Integer_Type, Integer_Type>**) mmap(nullptr, (tiling->rank_nrowgrps * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        
        struct Segment<Weight, Integer_Type, char>* s_i = new struct Segment<Weight, Integer_Type, char>(i_sizes[i], all_rowgrps_thread_sockets[i]);
        I2[i] = s_i;
        struct Segment<Weight, Integer_Type, Integer_Type>* s_iv = new struct Segment<Weight, Integer_Type, Integer_Type>(i_sizes[i], all_rowgrps_thread_sockets[i]);
        IV2[i] = s_iv;
    }
    

    filter_vertices(_ROWS_);
        
    std::vector<Integer_Type> thread_sockets(num_owned_segments);    
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    std::vector<Integer_Type> rowgrp_nnz_rows_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_rows[j];
        rowgrp_nnz_rows_sizes[j] = nnz_row_sizes_loc[io];
    }
    
    
    
    rowgrp_nnz_rows2 = (struct Segment<Weight, Integer_Type, Integer_Type>**) mmap(nullptr, (num_owned_segments * sizeof(struct  Segment<Weight, Integer_Type, Integer_Type>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        struct Segment<Weight, Integer_Type, Integer_Type>* r_i = new struct Segment<Weight, Integer_Type, Integer_Type>(rowgrp_nnz_rows_sizes[i], thread_sockets[i]);
        rowgrp_nnz_rows2[i] = r_i;
    }
    for(int32_t j = 0; j < num_owned_segments; j++) {      
        uint32_t io = accu_segment_rows[j];
        auto* i_data = (char*) I2[io]->data;
        auto* rgj_data = (Integer_Type*) rowgrp_nnz_rows2[j]->data;
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i]) {
                rgj_data[k] = i;
                k++;
            }
        }    
    }

    
            
    std::vector<Integer_Type> j_sizes(tiling->rank_ncolgrps, tile_width);
    int num_colgrps_per_thread = tiling->rank_ncolgrps / num_owned_segments;
    assert((num_colgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_ncolgrps);
    std::vector<Integer_Type> all_colgrps_thread_sockets(tiling->rank_ncolgrps);    
    for(int i = 0; i < num_colgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_colgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    
    J2 = (struct Segment<Weight, Integer_Type, char>**) mmap(nullptr, (tiling->rank_ncolgrps * sizeof(struct Segment<Weight, Integer_Type, char>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    JV2 = (struct Segment<Weight, Integer_Type, Integer_Type>**) mmap(nullptr, (tiling->rank_ncolgrps * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        
        struct Segment<Weight, Integer_Type, char>* s_j = new struct Segment<Weight, Integer_Type, char>(j_sizes[i], all_colgrps_thread_sockets[i]);
        J2[i] = s_j;
        struct Segment<Weight, Integer_Type, Integer_Type>* s_jv = new struct Segment<Weight, Integer_Type, Integer_Type>(j_sizes[i], all_colgrps_thread_sockets[i]);
        JV2[i] = s_jv;
    }
         

    filter_vertices(_COLS_);
    
    std::vector<Integer_Type> colgrp_nnz_cols_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_cols[j];
        colgrp_nnz_cols_sizes[j] = nnz_col_sizes_loc[io];
    }
    
    colgrp_nnz_cols2 = (struct Segment<Weight, Integer_Type, Integer_Type>**) mmap(nullptr, (num_owned_segments * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*)), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        struct Segment<Weight, Integer_Type, Integer_Type>* c_j = new struct Segment<Weight, Integer_Type, Integer_Type>(colgrp_nnz_cols_sizes[i], thread_sockets[i]);
        colgrp_nnz_cols2[i] = c_j;
    }
    
    for(int32_t j = 0; j < num_owned_segments; j++) { 
        uint32_t jo = accu_segment_cols[j];    
        auto* j_data = (char*) J2[jo]->data;
        auto* cgj_data = (Integer_Type*) colgrp_nnz_cols2[j]->data;
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_width; i++) {
            if(j_data[i]) {
                cgj_data[k] = i;
                k++;
            }
        }    
    }
    */    

    

    /*
    std::vector<Integer_Type> i_sizes(tiling->rank_nrowgrps, tile_height);
    int num_rowgrps_per_thread = tiling->rank_nrowgrps / num_owned_segments;
    assert((num_rowgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_nrowgrps);
    std::vector<Integer_Type> all_rowgrps_thread_sockets(tiling->rank_nrowgrps);    
    for(int i = 0; i < num_rowgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_rowgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    
    I1.resize(tiling->rank_nrowgrps);
    IV1.resize(tiling->rank_nrowgrps);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        Segment<Weight, Integer_Type, char>* s_i = new Segment<Weight, Integer_Type, char>(i_sizes[i], all_rowgrps_thread_sockets[i]);
        I1[i] = s_i;
        Segment<Weight, Integer_Type, Integer_Type>* s_iv = new Segment<Weight, Integer_Type, Integer_Type>(i_sizes[i], all_rowgrps_thread_sockets[i]);
        IV1[i] = s_iv;
    }
    filter_vertices(_ROWS_);
    
    
    
    
    std::vector<Integer_Type> thread_sockets(num_owned_segments);    
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    std::vector<Integer_Type> rowgrp_nnz_rows_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_rows[j];
        rowgrp_nnz_rows_sizes[j] = nnz_row_sizes_loc[io];
    }
    
    
    
    rowgrp_nnz_rows1.resize(num_owned_segments);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        Segment<Weight, Integer_Type, Integer_Type>* r_i = new Segment<Weight, Integer_Type, Integer_Type>(rowgrp_nnz_rows_sizes[i], thread_sockets[i]);
        rowgrp_nnz_rows1[i] = r_i;
    }
    for(int32_t j = 0; j < num_owned_segments; j++) {      
        uint32_t io = accu_segment_rows[j];
        auto* i_data = (char*) I1[io]->data;
        auto* rgj_data = (Integer_Type*) rowgrp_nnz_rows1[j]->data;
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i]) {
                rgj_data[k] = i;
                k++;
            }
        }    
    }
    
    
            
    std::vector<Integer_Type> j_sizes(tiling->rank_ncolgrps, tile_width);
    int num_colgrps_per_thread = tiling->rank_ncolgrps / num_owned_segments;
    assert((num_colgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_ncolgrps);
    std::vector<Integer_Type> all_colgrps_thread_sockets(tiling->rank_ncolgrps);    
    for(int i = 0; i < num_colgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_colgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    
    
    J1.resize(tiling->rank_ncolgrps);
    JV1.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        Segment<Weight, Integer_Type, char>* s_j = new Segment<Weight, Integer_Type, char>(j_sizes[i], all_colgrps_thread_sockets[i]);
        J1[i] = s_j;
        Segment<Weight, Integer_Type, Integer_Type>* s_jv = new Segment<Weight, Integer_Type, Integer_Type>(j_sizes[i], all_colgrps_thread_sockets[i]);
        JV1[i] = s_jv;
    }
    filter_vertices(_COLS_);
    
    std::vector<Integer_Type> colgrp_nnz_cols_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_cols[j];
        colgrp_nnz_cols_sizes[j] = nnz_col_sizes_loc[io];
    }
    
    colgrp_nnz_cols1.resize(num_owned_segments);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        Segment<Weight, Integer_Type, Integer_Type>* c_j = new Segment<Weight, Integer_Type, Integer_Type>(colgrp_nnz_cols_sizes[i], thread_sockets[i]);
        colgrp_nnz_cols1[i] = c_j;
    }
    
    for(int32_t j = 0; j < num_owned_segments; j++) { 
        uint32_t jo = accu_segment_cols[j];    
        auto* j_data = (char*) J1[jo]->data;
        auto* cgj_data = (Integer_Type*) colgrp_nnz_cols1[j]->data;
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_width; i++) {
            if(j_data[i]) {
                cgj_data[k] = i;
                k++;
            }
        }    
    }
    */    
    

    
    

    
    std::vector<Integer_Type> i_sizes(tiling->rank_nrowgrps, tile_height);
    
    int num_rowgrps_per_thread = tiling->rank_nrowgrps / num_owned_segments;
    assert((num_rowgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_nrowgrps);
    std::vector<Integer_Type> all_rowgrps_thread_sockets(tiling->rank_nrowgrps);    
    for(int i = 0; i < num_rowgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_rowgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    
    I = new Vector<Weight, Integer_Type, char>(i_sizes, all_rowgrps_thread_sockets);
    IV = new Vector<Weight, Integer_Type, Integer_Type>(i_sizes, all_rowgrps_thread_sockets);
    

    filter_vertices(_ROWS_);
    
    std::vector<Integer_Type> thread_sockets(num_owned_segments);    
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    std::vector<Integer_Type> rowgrp_nnz_rows_sizes(num_owned_segments);
    //std::vector<int32_t> rowgrp_nnz_rows_segments(num_owned_segments);
    //std::iota(rowgrp_nnz_rows_segments.begin(), rowgrp_nnz_rows_segments.end(), 0);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_rows[j];
        rowgrp_nnz_rows_sizes[j] = nnz_row_sizes_loc[io];
    }
    
    rowgrp_nnz_rows = new Vector<Weight, Integer_Type, Integer_Type>(rowgrp_nnz_rows_sizes, thread_sockets);
    for(int32_t j = 0; j < num_owned_segments; j++) {            
        uint32_t io = accu_segment_rows[j];
        auto* i_data = (char*) I->data[io];
        Integer_Type k = 0;
        auto* rgj_data = (Integer_Type*) rowgrp_nnz_rows->data[j];
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i]) {
                rgj_data[k] = i;
                k++;
            }
        }    
    }
    
    std::vector<Integer_Type> j_sizes(tiling->rank_ncolgrps, tile_width);
    int num_colgrps_per_thread = tiling->rank_ncolgrps / num_owned_segments;
    assert((num_colgrps_per_thread * Env::nthreads) == (int32_t) tiling->rank_ncolgrps);
    std::vector<Integer_Type> all_colgrps_thread_sockets(tiling->rank_ncolgrps);    
    for(int i = 0; i < num_colgrps_per_thread; i++) {
        for(int j = 0; j < Env::nthreads; j++) {
            int k = j + (i * Env::nthreads);
            all_colgrps_thread_sockets[k] = Env::socket_of_thread(j);
        }
    }
    J = new Vector<Weight, Integer_Type, char>(j_sizes, all_colgrps_thread_sockets);
    JV = new Vector<Weight, Integer_Type, Integer_Type>(j_sizes, all_colgrps_thread_sockets);
    filter_vertices(_COLS_);
    std::vector<Integer_Type> colgrp_nnz_cols_sizes(num_owned_segments);
    //std::vector<int32_t> colgrp_nnz_cols_segments(num_owned_segments);
    //std::iota(colgrp_nnz_cols_segments.begin(), colgrp_nnz_cols_segments.end(), 0);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_cols[j];
        colgrp_nnz_cols_sizes[j] = nnz_col_sizes_loc[io];
    }
    colgrp_nnz_cols = new Vector<Weight, Integer_Type, Integer_Type>(colgrp_nnz_cols_sizes, thread_sockets);
    for(int32_t j = 0; j < num_owned_segments; j++) {            
        uint32_t jo = accu_segment_cols[j];
        auto* j_data = (char*) J->data[jo];
        Integer_Type k = 0;
        auto* cgj_data = (Integer_Type*) colgrp_nnz_cols->data[j];
        for(Integer_Type i = 0; i < tile_width; i++) {
            if(j_data[i]) {
                cgj_data[k] = i;
                k++;
            }
        }
    }
    
    
    
    

/*    
    I.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        I[i].resize(tile_height);
    IV.resize(tiling->rank_nrowgrps);
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++)
        IV[i].resize(tile_height);
    filter_vertices(_ROWS_);
    rowgrp_nnz_rows_t.resize(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {            
        uint32_t io = accu_segment_rows[j];
        auto& i_data = I[io];
        rowgrp_nnz_rows_t[j].resize(nnz_row_sizes_loc[io]);
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i]) {
                rowgrp_nnz_rows_t[j][k] = i;
                k++;
            }
        }    
    }
    
    J.resize(tiling->rank_ncolgrps);
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        J[i].resize(tile_width);
    JV.resize(tiling->rank_ncolgrps);
    for (uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        JV[i].resize(tile_width);    
    filter_vertices(_COLS_);
    colgrp_nnz_cols_t.resize(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {            
        uint32_t jo = accu_segment_cols[j];
        auto& j_data = J[jo];
        colgrp_nnz_cols_t[j].resize(nnz_col_sizes_loc[jo]);
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_width; i++) {
            if(j_data[i]) {
                colgrp_nnz_cols_t[j][k] = i;
                k++;
            }
        }    
    }    
*/    
}   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter_vertices(Filtering_type filtering_type_) {
    //std::vector<std::vector<char>> *K;
    //std::vector<std::vector<Integer_Type>> *KV;
    Vector<Weight, Integer_Type, char>* K;
    Vector<Weight, Integer_Type, Integer_Type>* KV;
    //std::vector<Segment<Weight, Integer_Type, char>*> K;
    //std::vector<Segment<Weight, Integer_Type, Integer_Type>*> KV;
    //struct Segment<Weight, Integer_Type, char>** K;
    //struct Segment<Weight, Integer_Type, Integer_Type>** KV;
    uint32_t rank_nrowgrps_, rank_ncolgrps_;
    uint32_t rowgrp_nranks_;
    Integer_Type tile_length;
    std::vector<int32_t> local_row_segments_;
    std::vector<int32_t> all_rowgrp_ranks_accu_seg_;
    std::vector<uint32_t> local_tiles_row_order_;
    int32_t accu_segment_rg_, accu_segment_row_;
    std::vector<int32_t> accu_segment_rows_;
    std::vector<int32_t> follower_rowgrp_ranks_; 
    std::vector<int32_t> follower_rowgrp_ranks_accu_seg_;
    std::vector<Integer_Type> nnz_sizes_all, nnz_sizes_loc;
    std::vector<int32_t> leader_ranks_, owned_segments_;
    if(filtering_type_ == _ROWS_) {
        //K = &I;
        //KV = &IV;
        K = I;
        KV = IV;
        //K = I1;
        //KV = IV1;
        //K = I2;
        //KV = IV2;
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
        accu_segment_rows_ = accu_segment_rows;
        leader_ranks_ = leader_ranks_row;
        owned_segments_ = owned_segments_row;
    }
    else if(filtering_type_ == _COLS_) {
        //K = &J;
        //KV = &JV;
        K = J;
        KV = JV;
        //K = J1;
        //KV = JV1;
        //K = J2;
        //KV = JV2;
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
        accu_segment_rows_ = accu_segment_cols;
        leader_ranks_ = leader_ranks_col;
        owned_segments_ = owned_segments_col;
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
        if(leader_ranks[local_row_segments_[j]] == Env::rank) {
            
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
        
        auto& f_data = F[fi][fo];
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
    
    nnz_sizes_all.resize(Env::nsegments);    
    std::vector<Integer_Type> nnz_sizes_loc_val_temp;
    std::vector<Integer_Type> nnz_sizes_loc_pos_temp;
    for(int32_t k = 0; k < num_owned_segments; k++) {
        fi = accu_segment_rows_[k];
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
        nnz_sizes_all[owned_segments[k]] = nnz_local;
        nnz_sizes_loc_val_temp.push_back(nnz_local);
        nnz_sizes_loc_pos_temp.push_back(owned_segments[k]);
    }
    
    std::vector<Integer_Type> nnz_sizes_all_val_temp(Env::nsegments);
    std::vector<Integer_Type> nnz_sizes_all_pos_temp(Env::nsegments);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        int32_t j = i + (Env::rank * num_owned_segments);
        nnz_sizes_all_val_temp[j] = nnz_sizes_loc_val_temp[i];
        nnz_sizes_all_pos_temp[j] = nnz_sizes_loc_pos_temp[i];
    }
    
    for (int32_t i = 0; i < Env::nranks; i++) {
        int32_t j = Env::rank * num_owned_segments;
        int32_t k = i * num_owned_segments;
        if (i != Env::rank) {
            MPI_Sendrecv(&nnz_sizes_all_val_temp[j], num_owned_segments, TYPE_INT, i, 0, &nnz_sizes_all_val_temp[k], num_owned_segments, TYPE_INT, 
                                                                         i, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&nnz_sizes_all_pos_temp[j], num_owned_segments, TYPE_INT, i, 0, &nnz_sizes_all_pos_temp[k], num_owned_segments, TYPE_INT, 
                                                                         i, 0, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for(int32_t i = 0; i <  Env::nsegments; i++)  {
        nnz_sizes_all[nnz_sizes_all_pos_temp[i]] = nnz_sizes_all_val_temp[i];
    }
   
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        nnz_sizes_loc.push_back(nnz_sizes_all[this_segment]);
    }
    
    for(int32_t k = 0; k < num_owned_segments; k++) { 
        if(nnz_sizes_all[accu_segment_rows_[k]]) {
            fi = accu_segment_rows_[k];
            fo = accu_segment_rg_;
            auto& f_data = F[fi][fo]; 
            Integer_Type f_nitems = tile_length;
            
            uint32_t ko = accu_segment_rows_[k];  
            //auto* kj_data = (char*) K[ko]->data;
            //auto* kvj_data = (Integer_Type*) KV[ko]->data;
            //auto* kj_data = (char*) K[ko]->data;
            //auto* kvj_data = (Integer_Type *) KV[ko]->data;
            auto* kj_data = (char*) K->data[ko];
            auto* kvj_data = (Integer_Type *) KV->data[ko];
            //auto &kj_data =  (*K)[ko];
            //auto &kvj_data = (*KV)[ko];
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
            assert(j == nnz_sizes_all[owned_segments[k]]);
        }
    }
    
    for(uint32_t j = 0; j < rank_nrowgrps_; j++) {
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        //auto* kj_data = (char*) K[j]->data;
        //auto* kj_data = (char*) K[j]->data;
        auto* kj_data = (char*) K->data[j];
        //auto &kj_data = (*K)[j];
        if(Env::rank == leader) {
            for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++) {
                follower = follower_rowgrp_ranks_[i];
                MPI_Isend(kj_data, tile_length, TYPE_CHAR, follower, this_segment, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Irecv(kj_data, tile_length, TYPE_CHAR, leader, this_segment, Env::MPI_WORLD, &request);
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
        //auto &kvj_data = (*KV)[j];
        auto* kvj_data = (Integer_Type*) KV->data[j];
        //auto* kvj_data = (Integer_Type*) KV[j]->data;
        //auto* kvj_data = (Integer_Type*) KV[j]->data;
        if(Env::rank == leader) {
            for(uint32_t i = 0; i < rowgrp_nranks_ - 1; i++) {
                follower = follower_rowgrp_ranks_[i];
                MPI_Isend(kvj_data, tile_length, TYPE_INT, follower, this_segment, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Irecv(kvj_data, tile_length, TYPE_INT, leader, this_segment, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    } 
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    out_requests.clear(); 
    
    for (uint32_t j = 0; j < rank_nrowgrps_; j++) {
        //auto* kj_data = (char*) K[j]->data;
        //auto* kvj_data = (Integer_Type*) KV[j]->data;
        //auto* kj_data = (char*) K[j]->data;
        //auto* kvj_data = (Integer_Type*) KV[j]->data;
        auto* kj_data = (char*) K->data[j];
        auto* kvj_data = (Integer_Type*) KV->data[j];
        //auto &kj_data = (*K)[j];
        //auto &kvj_data = (*KV)[j];
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
        this_segment = local_row_segments_[j];
        leader = leader_ranks[this_segment];
        if(Env::rank == leader) {
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
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression() {
    if(compression_type == _TCSC_){
        if(Env::is_master)
            printf("INFO(rank=%d): Edge compression: Triply Compressed Sparse Column (TCSC)\n", Env::rank);
        init_tcsc();
    }
    else {
        fprintf(stderr, "ERROR(rank=%d): Edge compression: Invalid compression type\n", Env::rank);
        Env::exit(1);
    }
    Env::barrier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc() {
    //Env::barrier();
    //Env::exit(0);
    //int nthreads = Env::nthreads;
    std::vector<std::thread> threads;
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_threaded, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_threaded(int tid) {
    int ret = Env::set_thread_affinity(tid);
    int cid = sched_getcpu();
    int sid =  Env::socket_of_cpu(cid);
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        yi = tile.ith;
        Integer_Type c_nitems = nnz_col_sizes_loc[xi];
        Integer_Type r_nitems = nnz_row_sizes_loc[yi];
        auto* i_data = (char*) I->data[yi];
        auto* iv_data = (Integer_Type*) IV->data[yi];
        auto* j_data = (char*) J->data[xi];
        auto* jv_data = (Integer_Type*) JV->data[xi];
        
        //auto* i_data = (char*) I1[yi]->data;
        //auto* iv_data = (Integer_Type*) IV1[yi]->data;
        //auto* j_data = (char*) J1[xi]->data;
        //auto* jv_data = (Integer_Type*) JV1[xi]->data;
        
        //auto* i_data = (char*) I2[yi]->data;
        //auto* iv_data = (Integer_Type*) IV2[yi]->data;
        //auto* j_data = (char*) J2[xi]->data;
        //auto* jv_data = (Integer_Type*) JV2[xi]->data;
        
        
        //auto& i_data = I[yi];
        //auto& iv_data = IV[yi];
        //auto& j_data = J[xi];
        //auto& jv_data = JV[xi];
        if(tile.nedges) {
            tile.compressor = new TCSC_BASE<Weight, Integer_Type>(tile.triples->size(), c_nitems, r_nitems, sid);
            tile.compressor->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data);
        }
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
        }
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_filter() {
    
    delete I;
    delete IV;
    delete J;
    delete JV;
    delete rowgrp_nnz_rows;
    delete colgrp_nnz_cols;
    
    /*
    for(auto* s: I1) {
        delete s;
    }
    for(auto* s: IV1) {
        delete s;
    }
    for(auto* s: J1) {
        delete s;
    }
    for(auto* s: JV1) {
        delete s;
    }
    for(auto* s: rowgrp_nnz_rows1) {
        delete s;
    }
    for(auto* s: colgrp_nnz_cols1) {
        delete s;
    }
    */
    
    /*
    uint64_t nbytes = 0;
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        delete I2[i];
        delete IV2[i];
    }
    nbytes = tiling->rank_nrowgrps * sizeof(struct Segment<Weight, Integer_Type, char>*);
    munmap(I2, nbytes);
    nbytes = tiling->rank_nrowgrps * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*);
    munmap(IV2, nbytes);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        delete J2[i];
        delete JV2[i];
    }
    nbytes = tiling->rank_nrowgrps * sizeof(struct Segment<Weight, Integer_Type, char>*);
    munmap(J2, nbytes);
    nbytes = tiling->rank_ncolgrps * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*);
    munmap(JV2, nbytes);
    for(int i = 0; i < num_owned_segments; i++) {
        delete rowgrp_nnz_rows2[i];
        delete colgrp_nnz_cols2[i];
    }
    nbytes = num_owned_segments * sizeof(struct Segment<Weight, Integer_Type, Integer_Type>*);
    munmap(rowgrp_nnz_rows2, nbytes);
    munmap(colgrp_nnz_cols2, nbytes);
    */
    
    /*
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        I[i].clear();
        I[i].shrink_to_fit();
        IV[i].clear();
        IV[i].shrink_to_fit();
    }
    I.clear();
    I.shrink_to_fit();
    IV.clear();
    IV.shrink_to_fit();
    
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        J[i].clear();
        J[i].shrink_to_fit();
        JV[i].clear();
        JV[i].shrink_to_fit();
    }
    J.clear();
    J.shrink_to_fit();
    JV.clear();
    JV.shrink_to_fit();   
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
#endif
