/*
 * matrix.hpp: Matrix implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef MATRIX_2dGP_HPP
#define MATRIX_2dGP_HPP

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_2dgp() {
    ColSort<Weight, Integer_Type> f_col;
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
        tile.npartitions = Env::nthreads;
        tile.compressor_t.resize(tile.npartitions);
        if(triples.size()) {
            std::vector<uint64_t> start(tile.npartitions);
            std::vector<uint64_t> end(tile.npartitions);
            tile.triples_t.resize(tile.npartitions);
            Integer_Type chunk_size = tile_height / tile.npartitions;
            Integer_Type offset = tile.rg * tile_height;
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                start[tid] = chunk_size * tid;
                end[tid]   = (tid == (tile.npartitions - 1)) ? tile_height : chunk_size * (tid+1);
                tile.triples_t[tid] = new std::vector<struct Triple<Weight, Integer_Type>>;
                for(auto& triple: triples) {
                    if(triple.row >= (offset + start[tid]) and triple.row < (offset + end[tid]))
                        tile.triples_t[tid]->push_back(triple); 
                }
                std::sort(tile.triples_t[tid]->begin(), tile.triples_t[tid]->end(), f_col);
            }
        }
    }
    
    std::vector<std::thread> threads;
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_threaded_2dgp, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
    del_triples_t();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_threaded_2dgp(int tid) {
    int ret = Env::set_thread_affinity(tid);
    int cid = sched_getcpu();
    int sid =  Env::socket_of_cpu(cid);
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        Integer_Type c_nitems = nnz_col_sizes_loc[xi];
        Integer_Type r_nitems = nnz_row_sizes_loc[yi];
        auto& i_data = I[yi];
        auto& iv_data = IV[yi];
        auto& j_data = J[xi];
        auto& jv_data = JV[xi];
        tile.compressor_t[tid] = new TCSC_BASE<Weight, Integer_Type>(tile.triples_t[tid]->size(), c_nitems, r_nitems, sid);
        tile.compressor_t[tid]->populate(tile.triples_t[tid], tile_height, tile_width, i_data, iv_data, j_data, jv_data);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
            yi++;
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf_2dgp() {
    ColSort<Weight, Integer_Type> f_col;
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        std::vector<struct Triple<Weight, Integer_Type>>& triples = *(tile.triples);
        tile.npartitions = Env::nthreads;
        tile.compressor_t.resize(tile.npartitions);
        if(triples.size()) {
            std::vector<uint64_t> start(tile.npartitions);
            std::vector<uint64_t> end(tile.npartitions);
            tile.triples_t.resize(tile.npartitions);
            Integer_Type chunk_size = tile_height / tile.npartitions;
            Integer_Type offset = tile.rg * tile_height;
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                start[tid] = chunk_size * tid;
                end[tid]   = (tid == (tile.npartitions - 1)) ? tile_height : chunk_size * (tid+1);
                tile.triples_t[tid] = new std::vector<struct Triple<Weight, Integer_Type>>;
                for(auto& triple: triples) {
                    if(triple.row >= (offset + start[tid]) and triple.row < (offset + end[tid]))
                        tile.triples_t[tid]->push_back(triple); 
                }
                std::sort(tile.triples_t[tid]->begin(), tile.triples_t[tid]->end(), f_col);
            }
        }
    }
    
    std::vector<std::thread> threads;
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf_threaded_2dgp, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
    del_triples_t();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::init_tcsc_cf_threaded_2dgp(int tid) {
    int ret = Env::set_thread_affinity(tid);
    int cid = sched_getcpu();
    int sid =  Env::socket_of_cpu(cid);
    uint32_t yi = 0, xi = 0, next_row = 0;
    for(uint32_t t: local_tiles_row_order) {
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
        auto& regular_cols_data = regular_cols[xi];
        auto& regular_cols_bv_data = regular_cols_bitvector[xi];
        auto& sink_cols_data = sink_cols[xi];
        auto& sink_cols_bv_data = sink_cols_bitvector[xi];
        tile.compressor_t[tid] = new TCSC_CF_BASE<Weight, Integer_Type>(tile.nedges, c_nitems, r_nitems, sid);
        tile.compressor_t[tid]->populate(tile.triples, tile_height, tile_width, i_data, iv_data, j_data, jv_data, regular_rows_data, regular_rows_bv_data, source_rows_data, source_rows_bv_data, regular_cols_data, regular_cols_bv_data, sink_cols_data, sink_cols_bv_data, sid);
        xi++;
        next_row = (((tile.nth + 1) % tiling->rank_ncolgrps) == 0);
        if(next_row) {
            xi = 0;
            yi++;
        }
    }    
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Matrix<Weight, Integer_Type, Fractional_Type>::del_triples_t() {
    Triple<Weight, Integer_Type> pair;
    for(uint32_t t: local_tiles_row_order) {
        pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        if(tile.nedges) {
            for(int32_t i = 0; i < tile.npartitions; i++) {
                tile.triples_t[i]->clear();
                tile.triples_t[i]->shrink_to_fit();
                delete tile.triples_t[i];
                tile.triples_t[i] = nullptr;
            }
        }
    }    
}

#endif
