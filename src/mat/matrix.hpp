/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include "mpi/types.hpp" 
#include "mat/tiling.hpp" 
#include "mat/hashers.hpp"

#include "ds/vector.hpp" 


enum Filtering_type
{
  _ROWS_,
  _COLS_
}; 

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Tile2D { 
    template <typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Matrix;
    std::vector<struct Triple<Weight, Integer_Type>> triples;
    struct Compressed_column<Weight, Integer_Type>* compressor = nullptr;
    uint32_t rg, cg; // Row group, Column group
    uint32_t ith, jth, nth; // ith row, jth column, nth local row order tile,
    uint32_t mth, kth; // mth local column order tile, and kth global tile
    int32_t rank;
    int32_t thread;
    int32_t thread_global;
    int32_t leader;
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

//template<typename Weight, typename Integer_Type, typename Fractional_Type>
//void Tile2D<Weight, Integer_Type, Fractional_Type>::allocate_triples() {
//    if(!triples)
        //triples = new std::vector<struct Triple<Weight, Integer_Type>>;
//}
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Tile2D<Weight, Integer_Type, Fractional_Type>::free_triples() {
    //if(triples) {
        triples.clear();
        triples.shrink_to_fit();
    //    delete triples;
    //    triples = nullptr;
    //}
}
 
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Matrix {
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__, typename Vertex_State, typename Vertex_Methods_Impl>
    friend class Vertex_Program;
    
    public:    
        Matrix(Integer_Type nrows_, Integer_Type ncols_, uint32_t ntiles_, bool directed_, bool transpose_, bool parallel_edges_,
               Tiling_type tiling_type_, Compression_type compression_type_, Hashing_type hashing_type_);
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
        Hashing_type hashing_type;
        ReversibleHasher *hasher = nullptr;
        
        std::vector<Integer_Type> rows_sizes;
        Integer_Type rows_size;
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_rows_values;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        Integer_Type nnz_cols_size_loc;

        char** I = nullptr;
        Integer_Type** IV = nullptr;
        char** J = nullptr;
        Integer_Type** JV = nullptr;
        Integer_Type** rowgrp_nnz_rows = nullptr;
        Integer_Type** colgrp_nnz_cols = nullptr;        
        
        std::vector<std::vector<Integer_Type>> regular_rows; 
        std::vector<std::vector<char>> regular_rows_bitvector; 
        std::vector<std::vector<Integer_Type>> source_rows;
        std::vector<std::vector<char>> source_rows_bitvector;
        std::vector<std::vector<Integer_Type>> regular_cols;
        std::vector<std::vector<char>> regular_cols_bitvector;
        std::vector<std::vector<Integer_Type>> sink_cols;
        std::vector<std::vector<char>> sink_cols_bitvector;
        
        Integer_Type** rowgrp_regular_rows;
        Integer_Type** rowgrp_source_rows;
        Integer_Type** colgrp_sink_cols;
        std::vector<struct blk<Integer_Type>> rowgrp_regular_rows_blks;
        std::vector<struct blk<Integer_Type>> rowgrp_source_rows_blks;
        std::vector<struct blk<Integer_Type>> colgrp_sink_cols_blks;
        
        std::vector<struct blk<Integer_Type>> I_blks;
        std::vector<struct blk<Integer_Type>> IV_blks;
        std::vector<struct blk<Integer_Type>> J_blks;
        std::vector<struct blk<Integer_Type>> JV_blks;
        std::vector<struct blk<Integer_Type>> rgs_blks;
        std::vector<struct blk<Integer_Type>> cgs_blks;
        
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
        //std::vector<int32_t> owned_segments_row, owned_segments_col;
        int32_t num_owned_segments;
        std::vector<int32_t> owned_segments_all;
        std::vector<int32_t> owned_segments_thread, accu_segments_rows_thread, accu_segments_cols_thread, tid_thread, rowgrp_owner_thread, colgrp_owner_thread;
        std::vector<std::vector<int32_t>> rowgrp_owner_thread_segments, colgrp_owner_thread_segments;
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
        void free_hasher();
        void init_matrix();
        void del_triples();
        void init_tiles();
        void init_threads();
        void closest(std::vector<struct Triple<Weight, Integer_Type>>& triples, int32_t npartitions, uint64_t chunk_size_, 
                     std::vector<uint64_t>& start, std::vector<uint64_t>& end);
        void init_compression();
        void init_tcsc();
        void init_tcsc_threaded(int tid);
        void classify_vertices();
        void init_tcsc_cf();
        void init_tcsc_cf_threaded(int tid);
        
        //void init_tcsc_2dgp();
        //void init_tcsc_threaded_2dgp(int tid);
        //void init_tcsc_cf_2dgp();
        //void init_tcsc_cf_threaded_2dgp(int tid);
        void del_triples_t();
;
        void del_compression();
        void del_filter();
        void del_classifier();
        void print(std::string element);
        void distribute();
        void balance();
        void init_filtering();
        void filter_vertices(Filtering_type filtering_type_);
};

//#include "mat/matrix_2dgp.hpp"

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Matrix<Weight, Integer_Type, Fractional_Type>::Matrix(Integer_Type nrows_, 
    Integer_Type ncols_, uint32_t ntiles_, bool directed_, bool transpose_, bool parallel_edges_, Tiling_type tiling_type_, 
    Compression_type compression_type_, Hashing_type hashing_type_) {
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
    tiling = new Tiling(Env::nranks, Env::nthreads, ntiles, nrowgrps, ncolgrps, tiling_type_);
    compression_type = compression_type_;
    
    // Initialize hashing
    hashing_type = hashing_type_;
    if (hashing_type == _NONE_)
        hasher = new NullHasher();
    else if (hashing_type == _BUCKET_) {
        //if(tiling_type_ == _2DGP_)
        //    hasher = new SimpleBucketHasher(nrows, Env::nranks);
        //else
            hasher = new SimpleBucketHasher(nrows, Env::nsegments);
    }
    
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
void Matrix<Weight, Integer_Type, Fractional_Type>::free_hasher() {
    delete hasher;
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
    tiles[pair.row][pair.col].triples.push_back(triple);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_matrix() {
    // Reserve the 2D vector of tiles. 
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++)
        tiles[i].resize(ncolgrps);
    
    int32_t gcd_r = std::gcd(tiling->rowgrp_nranks, tiling->colgrp_nranks);
    int32_t gcd_t = std::gcd(tiling->rowgrp_nthreads, tiling->colgrp_nthreads);
    bool old_formula = false;
    // Initialize tiles 
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            tile.rg = i;
            tile.cg = j;
            
            if(old_formula) {
                tile.rank = (i % tiling->colgrp_nranks) * tiling->rowgrp_nranks + (j % tiling->rowgrp_nranks);
                tile.thread = (i / tiling->colgrp_nranks) % Env::nthreads;
                tile.rank_cg = i % tiling->colgrp_nranks;
            }
            else {
            
            /*
            tile.rank = (((i % tiling->colgrp_nthreads) * tiling->rowgrp_nthreads + (j % tiling->rowgrp_nthreads)) 
                      + ((i / (tiling->nrowgrps/gcd_t)) * (tiling->thread_nrowgrps))) % Env::nranks;
                      
            tile.thread = (i / tiling->colgrp_nranks) % Env::nthreads;
            */

            
            tile.thread_global = (((i % tiling->colgrp_nthreads) * tiling->rowgrp_nthreads + (j % tiling->rowgrp_nthreads)) 
                               + ((i / (tiling->nrowgrps/gcd_t)) * (tiling->thread_nrowgrps))) % (Env::nranks * Env::nthreads);
            tile.rank = tile.thread_global % Env::nranks;
            tile.thread = tile.thread_global / Env::nranks;      
            tile.rank_cg = tile.rank / tiling->rowgrp_nranks;
            }

            if(tiling->tiling_type == Tiling_type::_NUMA_) {
                tile.rank = Env::ranks[tile.rank];
            }
            
            tile.ith = tile.rg / tiling->colgrp_nranks; 
            tile.jth = tile.cg / tiling->rowgrp_nranks;
            
            tile.rank_rg = j % tiling->rowgrp_nranks;
            //tile.rank_cg = tile.rank / tiling->colgrp_nranks;
            
            
            tile.leader_rank_rg = i;
            tile.leader_rank_cg = j;
            
            tile.leader_rank_rg_rg = i;
            tile.leader_rank_cg_cg = j;
            
            tile.nth = (tile.ith * tiling->rank_ncolgrps) + tile.jth;
            tile.mth = (tile.jth * tiling->rank_nrowgrps) + tile.ith;
                        
            //tile.allocate_triples();
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
    //if(tiling->tiling_type == Tiling_type::_2DGP_)
    //    assert(num_owned_segments == 1);
    //else {
    if(numa_available() == -1) {
        Env::nthreads = 1;
        omp_set_num_threads(Env::nthreads);
    }
    assert(num_owned_segments == Env::nthreads);  
    //}
    
    
    //if(!Env::rank) {
    //    print("rank_cg");
    //}
    
    
if(old_formula) {  
    for (uint32_t i = 0; i < nrowgrps; i++) {
        
        for (uint32_t j = i; j < nrowgrps; j++) { 
            if(counts[tiles[j][i].rank] < num_owned_segments) {        
                counts[tiles[j][i].rank]++;
                if(i != j)
                    std::swap(tiles[i], tiles[j]);
                break;
            }
        }
        
        leader_ranks[i] = tiles[i][i].rank;
        leader_ranks_rg[i] = tiles[i][i].rank_rg;
        leader_ranks_cg[i] = tiles[i][i].rank_cg;
    }
}    
else {
    /*
    std::vector<std::vector<int>> ranks_column_groups(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        ranks_column_groups[i].resize(tiling->colgrp_nranks);
    
    if(!Env::rank) {
        for (int r = 0; r < Env::nranks; r++) {
            printf("%d %d\n", r, r/tiling->rowgrp_nranks);
        }
    }
    
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) { 
            tiles[i][j].rank_cg = 
        }
    }
    */
   for (uint32_t i = 0; i < nrowgrps; i++) {
        counts[tiles[i][i].rank]++;
        leader_ranks[i] = tiles[i][i].rank;
        leader_ranks_rg[i] = tiles[i][i].rank_rg;
        leader_ranks_cg[i] = tiles[i][i].rank_cg;
    }
}
    
   //if(!Env::rank) {
    //    print("rank_cg");
    //    print("rank");
    //}
    
    //Env::barrier();
    //Env::exit(0);
    
    
    
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
    
    
    

    //Env::barrier();
    //Env::exit(0);
    
    
    

    /* Spilitting communicator among row/col groups and creating
       the methadata required for processing them in vertex program */

    if(not Env::get_init_status()) {
        uint32_t ncommunicators = (tiling->rank_nrowgrps >= tiling->rank_ncolgrps) ? tiling->rank_nrowgrps : tiling->rank_ncolgrps;
        //printf("1.ncommunicators=%d\n", ncommunicators);
        Env::rowgrps_init(all_rowgrp_ranks, tiling->rowgrp_nranks, ncommunicators);
        //Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks);
        ncommunicators = (tiling->rank_ncolgrps >= (uint32_t) Env::nthreads) ? tiling->rank_ncolgrps : Env::nthreads;
        //printf("2.ncommunicators=%d\n", ncommunicators);
        Env::colgrps_init(all_colgrp_ranks, tiling->colgrp_nranks, ncommunicators);
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
    
    
    
    

    /* Distribute tiles among threads in a way that
       each thread is the owner of one row group
    */
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

    std::vector<int32_t> thread_nsegments(num_owned_segments);
    std::vector<std::vector<int32_t>> thread_segments(num_owned_segments);
    std::vector<std::vector<int32_t>> local_tiles(num_owned_segments);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        for(int32_t t: local_tiles_row_order_t[i]) {
            pair = tile_of_local_tile(t);
            local_tiles[i].push_back(pair.row);
        }
        local_tiles[i].erase(std::unique(local_tiles[i].begin(), local_tiles[i].end()), local_tiles[i].end());
        for(int32_t t: local_tiles[i]) {
            auto it = std::find(owned_segments.begin(), owned_segments.end(), t);
            if(it != owned_segments.end()){
                thread_nsegments[i]++;
                thread_segments[i].push_back(t);
            }           
        }
    }
    
    bool swap_tiles = false;
    if(not(std::equal(thread_nsegments.begin() + 1, thread_nsegments.end(), thread_nsegments.begin())))
        swap_tiles = true;
    
    if(swap_tiles) {
        for(int32_t i = 0; i < num_owned_segments; i++) {
            for(int32_t j = 0; j < num_owned_segments; j++) {
                if((thread_nsegments[i] > 1) and (thread_nsegments[j] == 0)) {
                    int32_t segment_i = thread_segments[i].back();
                    thread_segments[i].pop_back();
                    int32_t segment_j = -1;
                    for(int32_t t: local_tiles[j]) {
                        if (std::find(owned_segments.begin(), owned_segments.end(), t) == owned_segments.end()) {
                            segment_j = t;
                            break;
                        }
                    }
                    if(segment_j == -1) {
                        fprintf(stderr, "ERROR(rank=%d): Cannot find a segment to swap\n", Env::rank);
                        Env::exit(1);
                    }                        
                    
                    int32_t s_i = 0;
                    int32_t s_j = 0;
                    int32_t s_n = local_tiles_row_order_t[i].size();
                    while((s_i < s_n) and (s_j < s_n)) {
                        int t_i = local_tiles_row_order_t[i][s_i];
                        auto p_i = tile_of_local_tile(t_i);
                        int t_j = local_tiles_row_order_t[j][s_j];
                        auto p_j = tile_of_local_tile(t_j);
                        if((p_i.row == (uint32_t) segment_i) and (p_j.row == (uint32_t) segment_j)) {
                            local_tiles_row_order_t[i][s_i] = local_tiles_row_order_t[j][s_j];
                            local_tiles_row_order_t[j][s_j] = t_i;
                            s_i++;
                            s_j++;
                        }
                        else if(p_i.row != (uint32_t) segment_i)
                            s_i++;
                        else if(p_j.row != (uint32_t) segment_j)
                            s_j++;
                    }
                    
                    for(uint32_t k = 0; k < local_tiles[i].size(); k++) {
                        if(local_tiles[i][k] == segment_i) {
                            local_tiles[i][k] = segment_j;
                            break;
                        }
                    }
                    
                    for(uint32_t k = 0; k < local_tiles[j].size(); k++) {
                        if(local_tiles[j][k] == segment_j) {
                            local_tiles[j][k] = segment_i;
                            break;
                        }
                    }
                    
                    thread_nsegments[i]--;
                    thread_nsegments[j]++;
                }
            }
        }
    }
    if(not(std::equal(thread_nsegments.begin() + 1, thread_nsegments.end(), thread_nsegments.begin()))) {        
        fprintf(stderr, "ERROR(rank=%d): Failure configuring 2D tiling\n", Env::rank);
        Env::exit(1);
    }
    
    for(auto& t: local_tiles) {
        t.clear();
        t.shrink_to_fit();
    }
    local_tiles.clear();
    local_tiles.shrink_to_fit();
    
    std::vector<int32_t> places(num_owned_segments);
    int32_t j = 0;
    for(int32_t i = 0; i < num_owned_segments; i++) {
        for(int32_t t: local_tiles_row_order_t[i]) {
            pair = tile_of_local_tile(t);
            auto it = std::find(owned_segments.begin(), owned_segments.end(), pair.row);
            if(it != owned_segments.end()){
                auto idx = std::distance(owned_segments.begin(), it);
                places[i] = idx;
                break;
            }
        }
    }
   
    std::vector<int32_t> indices(num_owned_segments);
    std::iota(indices.begin(), indices.end(), 0);
    bool sort_tiles = false;
    for(int32_t i = 0; i < num_owned_segments; i++) {
        if(places[i] != indices[i]) {
            sort_tiles = true;
            break;
        }
    }
    
    if(sort_tiles) {
        std::vector<std::vector<int32_t>> tile_ids(num_owned_segments);
        for(int32_t i = 0; i < num_owned_segments; i++) {
            for(int32_t t: local_tiles_row_order_t[i]) {
                pair = tile_of_local_tile(t);
                auto it = std::find(owned_segments.begin(), owned_segments.end(), pair.row);
                if(it != owned_segments.end()){
                    auto idx = std::distance(owned_segments.begin(), it);
                    tile_ids[idx].push_back(t);
                }
            }
        }
        for(int32_t i = 0; i < num_owned_segments; i++) {
            int32_t k = 0;
            for(uint32_t j = 0; j < local_tiles_row_order_t[i].size(); j++) {    
                int32_t t = local_tiles_row_order_t[i][j];
                pair = tile_of_local_tile(t);
                auto it = std::find(owned_segments.begin(), owned_segments.end(), pair.row);
                if(it != owned_segments.end()){
                    local_tiles_row_order_t[i][j] = tile_ids[i][k];
                    k++;
                }
            }
        }
    }
    
    if(swap_tiles or sort_tiles) {
        //printf(">>>>%d %d\n", swap_tiles, sort_tiles);
        for(int32_t i = 0; i < num_owned_segments; i++) {
            std::sort(local_tiles_row_order_t[i].begin(), local_tiles_row_order_t[i].end());
        }
    }

    // Assignment of threads to row groups
    rowgrp_owner_thread.resize(tiling->rank_nrowgrps);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        for(int32_t t: local_tiles_row_order_t[i]) {
            pair = tile_of_local_tile(t);
            auto& tile = tiles[pair.row][pair.col];
            tile.thread = i;
            rowgrp_owner_thread[tile.ith] = i;
        }
    }
    rowgrp_owner_thread_segments.resize(num_owned_segments);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        rowgrp_owner_thread_segments[rowgrp_owner_thread[i]].push_back(i);
    }
    
    // Assignment of threads to column groups
    colgrp_owner_thread.resize(tiling->rank_ncolgrps);
    std::vector<std::vector<int32_t>> col_threads(tiling->rank_ncolgrps);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        for(int32_t t: local_tiles_col_order_t[i]) {
            pair = tile_of_local_tile(t);
            auto& tile = tiles[pair.row][pair.col];
            colgrp_owner_thread[tile.jth] = i;
        }
    }
    colgrp_owner_thread_segments.resize(num_owned_segments);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        colgrp_owner_thread_segments[colgrp_owner_thread[i]].push_back(i);
    }
    
    Env::barrier();
    if(Env::is_master) {
        printf("INFO(rank=%d): 2D tiling: %d x %d [nrows x ncols]\n", Env::rank, nrows, ncols);
        printf("INFO(rank=%d): 2D tiling: %d x %d [nrowgrps x ncolgrps]\n", Env::rank, nrowgrps, ncolgrps);
        printf("INFO(rank=%d): 2D tiling: %d x %d [nranks x nthreads] = %d = nsegments = nrowgrps = ncolgrps\n", Env::rank, Env::nranks, Env::nthreads, Env::nsegments);
        printf("INFO(rank=%d): 2D tiling: %d x %d [height x width]\n", Env::rank, tile_height, tile_width);
        printf("INFO(rank=%d): 2D tiling: %d x %d [rowgrp_nranks x colgrp_nranks]\n", Env::rank, tiling->rowgrp_nranks, tiling->colgrp_nranks);
        printf("INFO(rank=%d): 2D tiling: %d x %d [rank_nrowgrps x rank_ncolgrps]\n", Env::rank, tiling->rank_nrowgrps, tiling->rank_ncolgrps);
        printf("INFO(rank=%d): 2D tiling:         [nthreads (total)] = %d = nsegments = nrowgrps = ncolgrps\n", Env::rank, Env::nsegments);
        printf("INFO(rank=%d): 2D tiling: %d x %d [rowgrp_nthreads x colgrp_nthreads]\n", Env::rank, tiling->rowgrp_nthreads, tiling->colgrp_nthreads);
        printf("INFO(rank=%d): 2D tiling: %d x %d [thread_nrowgrps x thread_ncolgrps]\n", Env::rank, tiling->thread_nrowgrps, tiling->thread_ncolgrps);
    }
    print("rank");
    print("thread");
    //print("thread_global");
    
    ///Env::barrier(); 
    //Env::exit(0);
    
    
    owned_segments_thread.resize(Env::nsegments);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            MPI_Bcast(&tile.thread, 1, MPI_INT, tile.rank, Env::MPI_WORLD);
        }
        owned_segments_thread[i] = tiles[i][i].thread;
    }
    
    //if(!Env::rank)
    //    printf("\n");
    //print("thread");
    /*
        if(!Env::rank) {
        print("rank");
        
        printf("leader_ranks: ");
        for(auto l: leader_ranks)
            printf("%d ", l);
        printf("\n");
        
        printf("leader_ranks_rg: ");
        for(auto l: leader_ranks_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("leader_ranks_cg: ");
        for(auto l: leader_ranks_cg)
            printf("%d ", l);
        printf("\n");
        
        printf("owned_segments: ");
        for(auto l: owned_segments)
            printf("%d ", l);
        printf("\n");
        
        printf("owned_segments_all: ");
        for(auto l: owned_segments_all)
            printf("%d ", l);
        printf("\n");
        
        printf("counts: ");
        for(auto l: counts)
            printf("%d ", l);
        printf("\n");
        
        printf("local_tiles_row_order: ");
        for(auto l: local_tiles_row_order)
            printf("%d ", l);
        printf("\n");
        
        printf("local_tiles_col_order: ");
        for(auto l: local_tiles_col_order)
            printf("%d ", l);
        printf("\n");
        
        printf("local_col_segments: ");
        for(auto l: local_col_segments)
            printf("%d ", l);
        printf("\n");
        
        printf("local_row_segments: ");
        for(auto l: local_row_segments)
            printf("%d ", l);
        printf("\n");
        
        printf("all_rowgrp_ranks: ");
        for(auto l: all_rowgrp_ranks)
            printf("%d ", l);
        printf("\n");
        
        printf("all_rowgrp_ranks_accu_seg: ");
        for(auto l: all_rowgrp_ranks_accu_seg)
            printf("%d ", l);
        printf("\n");
        
        printf("all_rowgrp_ranks_rg: ");
        for(auto l: all_rowgrp_ranks_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("all_rowgrp_ranks_accu_seg_rg: ");
        for(auto l: all_rowgrp_ranks_accu_seg_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks: ");
        for(auto l: follower_rowgrp_ranks)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_accu_seg: ");
        for(auto l: follower_rowgrp_ranks_accu_seg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_rg: ");
        for(auto l: follower_rowgrp_ranks_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_accu_seg_rg: ");
        for(auto l: follower_rowgrp_ranks_accu_seg_rg)
            printf("%d ", l);
        printf("\n");
        
        
        printf("all_colgrp_ranks: ");
        for(auto l: all_colgrp_ranks)
            printf("%d ", l);
        printf("\n");
        
        printf("all_colgrp_ranks_accu_seg: ");
        for(auto l: all_colgrp_ranks_accu_seg)
            printf("%d ", l);
        printf("\n");
        
        printf("all_colgrp_ranks_cg: ");
        for(auto l: all_colgrp_ranks_cg)
            printf("%d ", l);
        printf("\n");
        
        printf("all_colgrp_ranks_accu_seg_cg: ");
        for(auto l: all_colgrp_ranks_accu_seg_cg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_colgrp_ranks: ");
        for(auto l: follower_colgrp_ranks)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_accu_seg: ");
        for(auto l: follower_rowgrp_ranks_accu_seg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_rg: ");
        for(auto l: follower_rowgrp_ranks_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("follower_rowgrp_ranks_accu_seg_rg: ");
        for(auto l: follower_rowgrp_ranks_accu_seg_rg)
            printf("%d ", l);
        printf("\n");
        
        printf("accu_segment_rows: ");
        for(auto l: accu_segment_rows)
            printf("%d ", l);
        printf("\n");
        
        printf("accu_segment_cols: ");
        for(auto l: accu_segment_cols)
            printf("%d ", l);
        printf("\n");
        
        
    }
   */ 
    
    
    
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
                if(element.compare("thread_global") == 0) 
                    printf("%02d ", tile.thread_global);
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
        printf("\n");
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
        std::vector<struct Triple<Weight, Integer_Type>>& triples = tile.triples;
        if(triples.size()) {
            std::sort(triples.begin(), triples.end(), f_col);
            /* remove parallel edges (duplicates), necessary for triangle couting */
            if(not parallel_edges) {
                auto last = std::unique(triples.begin(), triples.end(), f_comp);
                triples.erase(last, triples.end());
            }
        }
        tile.nedges = tile.triples.size();
    }
    //balance();

}


/* Inspired by LA3 code @
   https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/dist_matrix2d.hpp
*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::distribute() {
    //MPI_Barrier(MPI_COMM_WORLD);
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
            if(tile.triples.size() > 0)
                nedges_start_local += tile.triples.size();
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
                outbox.insert(outbox.end(), tile.triples.begin(), tile.triples.end());
                tile.free_triples();
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
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
                //std::vector<struct Triple<Weight, Integer_Type>>& triples = tile.triples;
                nedges_end_local += tile.triples.size();
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
    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::balance()
{
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<std::vector<uint64_t>> nedges_grid(Env::nranks);
    std::vector<uint64_t> rank_nedges(Env::nranks);
    std::vector<uint64_t> rowgrp_nedges(nrowgrps);
    std::vector<uint64_t> colgrp_nedges(ncolgrps);
    
    for(int32_t i = 0; i < Env::nranks; i++)
        nedges_grid[i].resize(tiling->rank_ntiles);
    
    uint32_t k = 0;
    for(uint32_t i = 0; i < nrowgrps; i++)
    {
        for(uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            if(Env::rank == (int32_t) tile.rank)
            {
                nedges_grid[Env::rank][k] = tile.nedges;
                k++;
            }
        }
    }

    for(int32_t r = 0; r < Env::nranks; r++)
    {
        if(r != Env::rank)
        {
            auto &out_edges = nedges_grid[Env::rank];
            auto &in_edges = nedges_grid[r];
            MPI_Sendrecv(out_edges.data(), out_edges.size(), MPI_UNSIGNED_LONG, r, Env::rank, 
                         in_edges.data(), in_edges.size(), MPI_UNSIGNED_LONG, r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(uint32_t i = 0; i < nrowgrps; i++)
    {
        for(uint32_t j = 0; j < ncolgrps; j++)  
        {
            auto &tile = tiles[i][j];
            if(Env::rank != tile.rank)
                tile.nedges = nedges_grid[tile.rank][tile.nth];
            
            rank_nedges[tile.rank] += tile.nedges;
            rowgrp_nedges[i] += tile.nedges;
            colgrp_nedges[j] += tile.nedges;
            nedges += tile.nedges;
        }
    }
    
    print("nedges");
    MPI_Barrier(MPI_COMM_WORLD); 
    if(!Env::rank)
    {   
        int32_t skip = 15;
        double imbalance_threshold = .2;
        double ratio = 0;
        uint32_t count = 0;
        printf("\nEdge balancing info (Not functional):\n");  
        printf("Edge balancing: Total number of edges = %lu\n", nedges);
        printf("Edge balancing: Balanced number of edges per ranks = %lu \n", nedges/Env::nranks);
        printf("Edge balancing: imbalance ratio per ranks [0-%d]\n", Env::nranks-1);
        printf("Edge balancing: ");
        for(int32_t r = 0; r < Env::nranks; r++)
        {
            ratio = (double) (rank_nedges[r] / (double) (nedges/Env::nranks));
            if(r < skip)
                printf("%2.2f ", ratio);
            if(fabs(ratio - 1) > imbalance_threshold)
                count++;
        }
        if(Env::nranks > skip)
            printf("...\n");
        else
            printf("\n");
        if(count)
        {
            printf("Edge balancing: Edge distribution among %d ranks are not balanced.\n", count);
        }
        count = 0;
        
        printf("Edge balancing: Imbalance ratio per rowgroups [0-%d]\n", nrowgrps-1);
        printf("Edge balancing: ");
        for(uint32_t i = 0; i < nrowgrps; i++)
        {
            ratio = (double) (rowgrp_nedges[i] / (double) (nedges/nrowgrps));
            if(i < (uint32_t) skip)
                printf("%2.2f ", ratio);
            if(fabs(ratio - 1) > imbalance_threshold)
                count++;
        }
        if(nrowgrps > (uint32_t) skip)
            printf("...\n");
        else
            printf("\n");
        if(count)
        {
            printf("Edge balancing: Edge distribution among %d rowgroups are not balanced.\n", count);
        }
        count = 0;
        
        printf("Edge balancing: Imbalance ratio per colgroups [0-%d]\n", ncolgrps-1);
        printf("Edge balancing: ");
        for(uint32_t j = 0; j < ncolgrps; j++)
        {
            ratio = (double) (colgrp_nedges[j] / (double) (nedges/ncolgrps));
            if(j < (uint32_t) skip)
                printf("%2.2f ", ratio);
            if(fabs(ratio - 1) > imbalance_threshold)
                count++;
        }
        if(ncolgrps > (uint32_t) skip)
            printf("...\n");
        else
            printf("\n");
        if(count)
        {
            printf("Edge balancing: Edge distribution among %d colgroups are not balanced.", count);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_filtering() {
    if(Env::is_master)
        printf("INFO(rank=%d): Vertex filtering: Filtering zero rows/cols\n", Env::rank);
    
    /* Filtering rows */
    std::vector<Integer_Type> i_sizes(tiling->rank_nrowgrps, tile_height);
    std::vector<int32_t> all_rowgrps_thread_sockets(tiling->rank_nrowgrps);    
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        all_rowgrps_thread_sockets[i] = Env::socket_of_thread(rowgrp_owner_thread[i]);
    }
    I_blks.resize(tiling->rank_nrowgrps);
    IV_blks.resize(tiling->rank_nrowgrps);
    for(uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        auto& i_blk = I_blks[i];
        i_blk.nitems = i_sizes[i];
        i_blk.socket_id = all_rowgrps_thread_sockets[i];
        auto& iv_blk = IV_blks[i];
        iv_blk.nitems = i_sizes[i];
        iv_blk.socket_id = all_rowgrps_thread_sockets[i];
    }
    allocate_numa_vector<Integer_Type, char>(&I, I_blks);
    allocate_numa_vector<Integer_Type, Integer_Type>(&IV, IV_blks);
    filter_vertices(_ROWS_);
    
    std::vector<int32_t> thread_sockets(num_owned_segments);    
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    std::vector<Integer_Type> rowgrp_nnz_rows_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t io = accu_segment_rows[j];
        rowgrp_nnz_rows_sizes[j] = nnz_row_sizes_loc[io];
    }
    rgs_blks.resize(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        auto& blk = rgs_blks[j];
        blk.nitems = rowgrp_nnz_rows_sizes[j];
        blk.socket_id = thread_sockets[j];
    }
    allocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_nnz_rows, rgs_blks);
    for(int32_t j = 0; j < num_owned_segments; j++) {      
        uint32_t io = accu_segment_rows[j];
        auto* i_data = (char*) I[io];
        auto* rgj_data = (Integer_Type*) rowgrp_nnz_rows[j];
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i]) {
                rgj_data[k] = i;
                k++;
            }
        }    
    }
   
    /* Filtering columns */
    std::vector<Integer_Type> j_sizes(tiling->rank_ncolgrps, tile_width);
    std::vector<int32_t> all_colgrps_thread_sockets(tiling->rank_ncolgrps);     
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        all_colgrps_thread_sockets[i] = Env::socket_of_thread(colgrp_owner_thread[i]);
    }
    J_blks.resize(tiling->rank_ncolgrps);
    JV_blks.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        auto& j_blk = J_blks[i];
        j_blk.nitems = j_sizes[i];
        j_blk.socket_id = all_colgrps_thread_sockets[i];
        auto& jv_blk = JV_blks[i];
        jv_blk.nitems = j_sizes[i];
        jv_blk.socket_id = all_colgrps_thread_sockets[i];
    }
    allocate_numa_vector<Integer_Type, char>(&J, J_blks);
    allocate_numa_vector<Integer_Type, Integer_Type>(&JV, JV_blks);
    filter_vertices(_COLS_);
    
    std::vector<Integer_Type> colgrp_nnz_cols_sizes(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        uint32_t jo = accu_segment_cols[j];
        colgrp_nnz_cols_sizes[j] = nnz_col_sizes_loc[jo];
    }
    cgs_blks.resize(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        auto& blk = cgs_blks[j];
        blk.nitems = colgrp_nnz_cols_sizes[j];
        blk.socket_id = thread_sockets[j];
    }
    allocate_numa_vector<Integer_Type, Integer_Type>(&colgrp_nnz_cols, cgs_blks);
    for(int32_t j = 0; j < num_owned_segments; j++) { 
        uint32_t jo = accu_segment_cols[j];    
        auto* j_data = (char*) J[jo];
        auto* cgj_data = (Integer_Type*) colgrp_nnz_cols[j];
        Integer_Type k = 0;
        for(Integer_Type i = 0; i < tile_width; i++) {
            if(j_data[i]) {
                cgj_data[k] = i;
                k++;
            }
        }    
    }
}   

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::filter_vertices(Filtering_type filtering_type_) {
    char** K;
    Integer_Type** KV;
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
        K = I;
        KV = IV;
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
        //owned_segments_ = owned_segments_row;
    }
    else if(filtering_type_ == _COLS_) {
        K = J;
        KV = JV;
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
        //owned_segments_ = owned_segments_col;
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
            for (auto& triple : tile.triples) {
                test(triple);
                auto pair1 = rebase(triple);
                if(!f_data[pair1.row])
                    f_data[pair1.row] = 1;
            }
        }
        else if(filtering_type_ == _COLS_) {
            for (auto& triple : tile.triples) {
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
            auto* kj_data = (char*) K[ko];
            auto* kvj_data = (Integer_Type*) KV[ko];
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
        auto* kj_data = (char*) K[j];
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
        auto* kvj_data = (Integer_Type*) KV[j];
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
        auto* kj_data = (char*) K[j];
        auto* kvj_data = (Integer_Type*) KV[j];
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
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_compression() {
    if(compression_type == _TCSC_){
        if(Env::is_master)
            printf("INFO(rank=%d): Edge compression: Triply Compressed Sparse Column (TCSC)\n", Env::rank);
        //if(tiling->tiling_type == _2DGP_)
        //    init_tcsc_2dgp();
        //else
            init_tcsc();
    }
    else if(compression_type == _TCSC_CF_){
        if(Env::is_master)
            printf("INFO(rank=%d): Edge compression: Triply Compressed Sparse Column (TCSC) - Computation Filtering\n", Env::rank);
        classify_vertices();
        //if(tiling->tiling_type == _2DGP_)
        //    init_tcsc_cf_2dgp();
        //else
            init_tcsc_cf();
        del_classifier();
    }
    else {
        fprintf(stderr, "ERROR(rank=%d): Edge compression: Invalid compression type\n", Env::rank);
        Env::exit(1);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc() {
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
        
        auto* i_data = (char*) I[yi];
        auto* iv_data = (Integer_Type*) IV[yi];
        auto* j_data = (char*) J[xi];
        auto* jv_data = (Integer_Type*) JV[xi];
        tile.compressor = new TCSC_BASE<Weight, Integer_Type>(tile.triples.size(), c_nitems, r_nitems, sid);
        tile.compressor->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::classify_vertices() {

    std::vector<Integer_Type> regular_rows_count(num_owned_segments);
    std::vector<Integer_Type> regular_cols_count(num_owned_segments);
    std::vector<Integer_Type> source_rows_count(num_owned_segments);
    std::vector<Integer_Type> sink_cols_count(num_owned_segments);
    
    for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t io = accu_segment_rows[k];
        auto& i_data = I[io];
        uint32_t jo = accu_segment_cols[k];
        auto& j_data = J[jo];  
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i] and j_data[i]) {
                regular_rows_count[k]++;
                regular_cols_count[k]++;
            }
            if(i_data[i] and !j_data[i])
                source_rows_count[k]++;
            if(!i_data[i] and j_data[i])
                sink_cols_count[k]++;
        }
    }

    regular_rows.resize(tiling->rank_nrowgrps);
    source_rows.resize(tiling->rank_nrowgrps);
    regular_cols.resize(tiling->rank_ncolgrps);
    sink_cols.resize(tiling->rank_ncolgrps);
    
    for(int32_t k = 0; k < num_owned_segments; k++) { 
        uint32_t io = accu_segment_rows[k];
        regular_rows[io].resize(regular_rows_count[k]);
        source_rows[io].resize(source_rows_count[k]);
        uint32_t jo = accu_segment_cols[k];
        regular_cols[jo].resize(regular_cols_count[k]);
        sink_cols[jo].resize(sink_cols_count[k]);
    }
    
    for(int32_t k = 0; k < num_owned_segments; k++) { 
        uint32_t io = accu_segment_rows[k];
        auto& i_data = I[io];
        uint32_t jo = accu_segment_cols[k];
        auto& j_data = J[jo];  
        auto& regular_row = regular_rows[io];
        auto& source_row = source_rows[io];
        auto& regular_col = regular_cols[jo];
        auto& sink_col = sink_cols[jo];
        Integer_Type i_regular = 0;
        Integer_Type i_source = 0;
        Integer_Type i_sink = 0;
        for(Integer_Type i = 0; i < tile_height; i++) {
            if(i_data[i] and j_data[i]) {
                regular_row[i_regular] = i;
                regular_col[i_regular] = i;
                i_regular++;
            }
            if(i_data[i] and !j_data[i]) {
                source_row[i_source] = i;
                i_source++;
            }
            if(!i_data[i] and j_data[i]) {
                sink_col[i_sink] = i;
                i_sink++;
            }
        }
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
    int32_t idx = 0;
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        row_group = local_row_segments[i];
        leader = leader_ranks[row_group];
        my_rank = Env::rank;
        if(leader == my_rank) {            
            for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++) {
                follower = follower_rowgrp_ranks[j];
                uint32_t io = accu_segment_rows[idx];
                auto& regular_row = regular_rows[io];
                nitems = regular_row.size();
                MPI_Send(&nitems, 1, TYPE_INT, follower, row_group, Env::MPI_WORLD);
                MPI_Isend(regular_row.data(), regular_row.size(), TYPE_INT, follower, row_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            idx++;
        }
        else {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, row_group, Env::MPI_WORLD, &status);
            auto& regular_row = regular_rows[i];
            regular_row.resize(nitems);
            MPI_Irecv(regular_row.data(), regular_row.size(), TYPE_INT, leader, row_group, Env::MPI_WORLD, &request);
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
    idx = 0;
    for (uint32_t i = 0; i < tiling->rank_nrowgrps; i++) {
        row_group = local_row_segments[i];
        leader = leader_ranks[row_group];
        my_rank = Env::rank;
        if(leader == my_rank) {            
            for(uint32_t j = 0; j < tiling->rowgrp_nranks - 1; j++) {
                follower = follower_rowgrp_ranks[j];
                uint32_t io = accu_segment_rows[idx];
                auto& source_row = source_rows[io];
                nitems = source_row.size();
                MPI_Send(&nitems, 1, TYPE_INT, follower, row_group, Env::MPI_WORLD);
                MPI_Isend(source_row.data(), source_row.size(), TYPE_INT, follower, row_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
            idx++;
        }
        else {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, row_group, Env::MPI_WORLD, &status);
            auto& source_row = source_rows[i];
            source_row.resize(nitems);
            MPI_Irecv(source_row.data(), source_row.size(), TYPE_INT, leader, row_group, Env::MPI_WORLD, &request);
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
    for(int32_t k = 0; k < num_owned_segments; k++) { 
        uint32_t jo = accu_segment_cols[k];
        auto& regular_col = regular_cols[jo];
        nitems = regular_col.size();
        col_group = local_col_segments[jo];
        for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++) {
            follower = follower_colgrp_ranks[i];
            MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
            MPI_Isend(regular_col.data(), regular_col.size(), TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }
    }
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks[col_group];
        my_rank = Env::rank;
        if(leader != my_rank) {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
            auto& regular_col = regular_cols[i];
            regular_col.resize(nitems);
            MPI_Irecv(regular_col.data(), regular_col.size(), TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    regular_cols_bitvector.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        regular_cols_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        for(Integer_Type j: regular_cols[i])
            regular_cols_bitvector[i][j] = 1;
    }     
    
    // Sink columns
    for(int32_t k = 0; k < num_owned_segments; k++) { 
        uint32_t jo = accu_segment_cols[k];
        auto& sink_col = sink_cols[jo];
        nitems = sink_col.size();
        col_group = local_col_segments[jo];
        for(uint32_t i = 0; i < tiling->colgrp_nranks - 1; i++) {
            follower = follower_colgrp_ranks[i];
            MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
            MPI_Isend(sink_col.data(), sink_col.size(), TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }
    }
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks[col_group];
        my_rank = Env::rank;
        if(leader != my_rank) {
            MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
            auto& sink_col = sink_cols[i];
            sink_col.resize(nitems);
            MPI_Irecv(sink_col.data(), sink_col.size(), TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
            in_requests.push_back(request);
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    sink_cols_bitvector.resize(tiling->rank_ncolgrps);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++)
        sink_cols_bitvector[i].resize(tile_height);
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        for(Integer_Type j: sink_cols[i])
            sink_cols_bitvector[i][j] = 1;
    }   
    
    // Auxilary vectors for vertex program
    std::vector<int32_t> thread_sockets(num_owned_segments);
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    std::vector<Integer_Type> rowgrp_regular_rows_sizes(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        uint32_t io = accu_segment_rows[k];
        auto regular_row = regular_rows[io];
        rowgrp_regular_rows_sizes[k] = regular_row.size();
    }
    rowgrp_regular_rows_blks.resize(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto& blk = rowgrp_regular_rows_blks[k];
        blk.nitems = rowgrp_regular_rows_sizes[k];
        blk.socket_id = thread_sockets[k];
    }
    allocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_regular_rows, rowgrp_regular_rows_blks);
    
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto* regular_row_data = (Integer_Type*) rowgrp_regular_rows[k];
        uint32_t io = accu_segment_rows[k];
        auto regular_row = regular_rows[io];
        Integer_Type nitems = regular_row.size();
        for(Integer_Type i = 0; i < nitems; i++) {
            regular_row_data[i] = regular_row[i];
        }
    }

    std::vector<Integer_Type> rowgrp_source_rows_sizes(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        uint32_t io = accu_segment_rows[k];
        auto source_row = source_rows[io];
        rowgrp_source_rows_sizes[k] = source_row.size();
    }
    rowgrp_source_rows_blks.resize(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto& blk = rowgrp_source_rows_blks[k];
        blk.nitems = rowgrp_source_rows_sizes[k];
        blk.socket_id = thread_sockets[k];
    }
    allocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_source_rows, rowgrp_source_rows_blks);
    
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto* source_row_data = (Integer_Type*) rowgrp_source_rows[k];
        uint32_t io = accu_segment_rows[k];
        auto source_row = source_rows[io];
        Integer_Type nitems = source_row.size();
        for(Integer_Type i = 0; i < nitems; i++) {
            source_row_data[i] = source_row[i];
        }
    }
    
    std::vector<Integer_Type> colgrp_sink_cols_sizes(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        uint32_t jo = accu_segment_cols[k];
        auto sink_col = sink_cols[jo];
        colgrp_sink_cols_sizes[k] = sink_col.size();
    }
    colgrp_sink_cols_blks.resize(num_owned_segments);
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto& blk = colgrp_sink_cols_blks[k];
        blk.nitems = colgrp_sink_cols_sizes[k];
        blk.socket_id = thread_sockets[k];
    }
    allocate_numa_vector<Integer_Type, Integer_Type>(&colgrp_sink_cols, colgrp_sink_cols_blks);
    
    for(int32_t k = 0; k < num_owned_segments; k++) {  
        auto* sink_col_data = (Integer_Type*) colgrp_sink_cols[k];
        uint32_t jo = accu_segment_cols[k];
        auto sink_col = sink_cols[jo];
        Integer_Type nitems = sink_col.size();
        for(Integer_Type i = 0; i < nitems; i++) {
            sink_col_data[i] = sink_col[i];
        }
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf() {
    std::vector<std::thread> threads;
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf_threaded, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf_threaded(int tid) {
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
        auto* i_data = (char*) I[yi];
        auto* iv_data = (Integer_Type*) IV[yi];
        auto* j_data = (char*) J[xi];
        auto* jv_data = (Integer_Type*) JV[xi];
        auto& regular_rows_data = regular_rows[yi];
        auto& regular_rows_bv_data = regular_rows_bitvector[yi];
        auto& source_rows_data = source_rows[yi];
        auto& source_rows_bv_data = source_rows_bitvector[yi];
        auto& regular_cols_data = regular_cols[xi];
        auto& regular_cols_bv_data = regular_cols_bitvector[xi];
        auto& sink_cols_data = sink_cols[xi];
        auto& sink_cols_bv_data = sink_cols_bitvector[xi];
        tile.compressor = new TCSC_CF_BASE<Weight, Integer_Type>(tile.nedges, c_nitems, r_nitems, sid);
        tile.compressor->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data, regular_rows_data, regular_rows_bv_data, source_rows_data, source_rows_bv_data, regular_cols_data, regular_cols_bv_data, sink_cols_data, sink_cols_bv_data, sid);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_filter() {
    deallocate_numa_vector<Integer_Type, char>(&I, I_blks);
    deallocate_numa_vector<Integer_Type, Integer_Type>(&IV, IV_blks);
    deallocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_nnz_rows, rgs_blks);
    deallocate_numa_vector<Integer_Type, char>(&J, J_blks);
    deallocate_numa_vector<Integer_Type, Integer_Type>(&JV, JV_blks);
    deallocate_numa_vector<Integer_Type, Integer_Type>(&colgrp_nnz_cols, cgs_blks);
    if(compression_type == _TCSC_CF_) {
        deallocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_regular_rows, rowgrp_regular_rows_blks);
        deallocate_numa_vector<Integer_Type, Integer_Type>(&rowgrp_source_rows, rowgrp_source_rows_blks);
        deallocate_numa_vector<Integer_Type, Integer_Type>(&colgrp_sink_cols, colgrp_sink_cols_blks);
    }
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
        regular_cols[i].clear();
        regular_cols[i].shrink_to_fit();
        regular_cols_bitvector[i].clear();
        regular_cols_bitvector[i].shrink_to_fit();
    }
    regular_cols.clear();
    regular_cols.shrink_to_fit();
    regular_cols_bitvector.clear();
    regular_cols_bitvector.shrink_to_fit();
    for(uint32_t i = 0; i < tiling->rank_ncolgrps; i++) {
        sink_cols[i].clear();
        sink_cols[i].shrink_to_fit();
        sink_cols_bitvector[i].clear();
        sink_cols_bitvector[i].shrink_to_fit();
    }
    sink_cols.clear();
    sink_cols.shrink_to_fit();
    sink_cols_bitvector.clear();
    sink_cols_bitvector.shrink_to_fit();
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
