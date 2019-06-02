/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_2DGP_HPP
#define VERTEX_PROGRAM_2DGP_HPP

#include "vp/vertex_program.hpp"

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::function_stationary_2dgp() {
    if(not already_initialized)
        init_stationary_2dgp();   
    do {
        bcast_stationary_2dgp();
        combine_2d_stationary_2dgp();
        apply_stationary_2dgp();
    } while(not has_converged_2dgp());
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_stationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    int tid = 0;
    init_vectors();
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    for(uint32_t i = 0; i < tile_height; i++) {
        Vertex_State& state = v_data[i]; 
        c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[tid]), state);
    }

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_stationary_2dgp() {      
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    int tid = 0;
    uint32_t xo = accu_segment_cols[tid];       
    auto* x_data = (Fractional_Type*) X[xo];
    const auto* JC = (Integer_Type*) colgrp_nnz_cols[tid];
    Integer_Type JC_nitems = nnz_col_sizes_loc[xo];
    auto* v_data = (Vertex_State*) V[tid];
    for(uint32_t j = 0; j < JC_nitems; j++) {
        Vertex_State& state = v_data[JC[j]];
        x_data[j] = Vertex_Methods.messenger(state);
    }

    MPI_Request request;
    int32_t leader, col_group;    
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {    
        col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];  
        auto* xj_data = (Fractional_Type*) X[i];
        Integer_Type xj_nitems = nnz_col_sizes_loc[i];
        MPI_Ibcast(xj_data, xj_nitems, TYPE_DOUBLE, leader, colgrps_communicators[tid], &request);
        out_requests_t[tid].push_back(request);
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Bcast", elapsed_time);
    bcast_time.push_back(elapsed_time);
    #endif
} 


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_stationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif

    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader;
    int32_t follower, my_rank, accu;
    bool communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    //for(uint32_t t: local_tiles_row_order_t[tid]) {
    for(uint32_t t: local_tiles_row_order) {        
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        tile_th = tile.nth;
        pair_idx = pair.row;
        yi = tile.ith;
        auto* y_data = (Fractional_Type*) Y[yi];
        Integer_Type y_nitems = nnz_row_sizes_loc[yi];
        const auto* x_data = (Fractional_Type*) X[xi];
        if(compression_type == _TCSC_) {
            #pragma omp parallel
            {   
                int tid = omp_get_thread_num();
                int ret = Env::set_thread_affinity(tid);
                uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                if(nnz) { 
                    #ifdef HAS_WEIGHT
                    Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                    #endif
                    Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                    Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                    Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
                    for(uint32_t j = 0; j < ncols; j++) {
                        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                            #ifdef HAS_WEIGHT
                            Vertex_Methods.combiner(y_data[IA[i]], x_data[j], A[i]);
                            #else
                            Vertex_Methods.combiner(y_data[IA[i]], x_data[j]);
                            #endif
                        }
                    }
                }
            }
        }            
        else {
            #pragma omp parallel
            {   
                int tid = omp_get_thread_num();
                int ret = Env::set_thread_affinity(tid);
                uint64_t nnz = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                
                if(nnz) { 
                    #ifdef HAS_WEIGHT
                    Integer_Type* A = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                    #endif
                    Integer_Type* IA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                    Integer_Type* JA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                    Integer_Type ncols = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
                    
                    if(num_iterations == 1) {
                        if(not converged) {
                            for(uint32_t j = 0; j < ncols; j++) {
                                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                    #ifdef HAS_WEIGHT
                                    Vertex_Methods.combiner(y_data[IA[i]], x_data[j], A[i]);
                                    #else
                                    Vertex_Methods.combiner(y_data[IA[i]], x_data[j]);
                                    #endif
                                }
                            }
                        }
                    }
                    else {   
                        Integer_Type l;
                        if(iteration == 0) {               
                            Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_REG_R_SNK_C;
                            if(NC_REG_R_SNK_C) {
                                Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_REG_R_SNK_C;
                                Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_REG_R_SNK_C; j++, k = k + 2) {
                                    l = JC_REG_R_SNK_C[j];
                                    for(uint32_t i = JA_REG_R_SNK_C[k]; i < JA_REG_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                        }
                        if(not converged) {
                            Integer_Type NC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_REG_R_REG_C;
                            if(NC_REG_R_REG_C) {
                                Integer_Type* JC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_REG_R_REG_C;
                                Integer_Type* JA_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_REG_C;
                                for(uint32_t j = 0, k = 0; j < NC_REG_R_REG_C; j++, k = k + 2) {
                                    l = JC_REG_R_REG_C[j];
                                    for(uint32_t i = JA_REG_R_REG_C[k]; i < JA_REG_R_REG_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                        }
                        else {
                            Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_REG_C;
                            if(NC_SRC_R_REG_C) {
                                Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_REG_C;
                                Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_REG_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_REG_C; j++, k = k + 2) {
                                    l = JC_SRC_R_REG_C[j];
                                    for(uint32_t i = JA_SRC_R_REG_C[k]; i < JA_SRC_R_REG_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                            
                            Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_SNK_C;
                            if(NC_SRC_R_REG_C) {
                                Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_SNK_C;
                                Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_SNK_C; j++, k = k + 2) {
                                    l = JC_SRC_R_SNK_C[j];
                                    for(uint32_t i = JA_SRC_R_SNK_C[k]; i < JA_SRC_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                        }
                    }
                }
            }
        }
        
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            int tid = 0;
            leader = tile.leader_rank_rg_rg;
            my_rank = Env::rank_rg;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                    follower = follower_rowgrp_ranks_rg[j];
                    auto* yj_data = (Fractional_Type*) Yt[tid][j];
                    Integer_Type yj_nitems = nnz_row_sizes_all[owned_segments[tid]];
                    MPI_Irecv(yj_data, yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[yi], &request);
                    in_requests_t[tid].push_back(request);
                }
            }
            else {
                MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[yi], &request);
                out_requests_t[tid].push_back(request);
            }
            xi = 0;
        }
    }
    
    int tid = 0;
    
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();

    yi  = accu_segment_rows[tid];
    auto* y_data = (Fractional_Type*) Y[yi];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
    auto* yj_data = (Fractional_Type*) Yt[tid][j];
        Integer_Type yj_nitems = nnz_row_sizes_all[owned_segments[tid]];
        for(uint32_t i = 0; i < yj_nitems; i++)
            Vertex_Methods.combiner(y_data[i], yj_data[i]);
    }
   
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();    
    
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Combine", elapsed_time);
    combine_time.push_back(elapsed_time);
    #endif    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_stationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    int tid = 0;
    uint32_t yi  = accu_segment_rows[tid];
    const auto* i_data = (char*) I[yi];
    const auto* iv_data = (Integer_Type*) IV[yi];
    auto* y_data = (Fractional_Type*) Y[yi];
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];

    if(computation_filtering) {
        if(not converged) {
            auto* regular_rows = (Integer_Type*) rowgrp_regular_rows[tid];
            Integer_Type reg_nitems = rowgrp_regular_rows_blks[tid].nitems;
            for(Integer_Type r = 0; r < reg_nitems; r++) {
                Integer_Type i = regular_rows[r];
                Vertex_State& state = v_data[i];
                Integer_Type j = iv_data[i];    
                c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
            }    
        }
        else {
            auto* source_rows = (Integer_Type*) rowgrp_source_rows[tid];
            Integer_Type src_nitems = rowgrp_source_rows_blks[tid].nitems;
            for(Integer_Type r = 0; r < src_nitems; r++) {
                Integer_Type i = source_rows[r];
                Vertex_State &state = v_data[i];
                Integer_Type j = iv_data[i];
                c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
            }
        }
    }
    else {
        for(uint32_t i = 0; i < tile_height; i++) {
            Vertex_State &state = v_data[i];
            Integer_Type j = iv_data[i];
            if(i_data[i]) {
                c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
            }
            else
                c_data[i] = Vertex_Methods.applicator(state);
        }   
    }
    
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Apply", elapsed_time);
    apply_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::function_nonstationary_2dgp() {
    
    if(not already_initialized)
        init_nonstationary_2dgp();    
    do {
        bcast_nonstationary_2dgp();
        combine_2d_nonstationary_2dgp();
        apply_nonstationary_2dgp();
    } while(not has_converged_2dgp());
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_nonstationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif   
    
    int tid = 0;
    init_vectors();
    // Activity vectors
    activity_statuses.resize(ncolgrps);
    msgs_activity_statuses.resize(rank_ncolgrps);
    accus_activity_statuses.resize(Env::nthreads, std::vector<Integer_Type>(rowgrp_nranks - 1));

    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    for(uint32_t i = 0; i < tile_height; i++) {
        Vertex_State& state = v_data[i]; 
        c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[tid]), state);
    }
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++) {
        auto* y_data = (Fractional_Type*) Y[i];
        Integer_Type y_nitems = nnz_row_sizes_loc[i];
        for(uint32_t j = 0; j < y_nitems; j++)
            y_data[j] = Vertex_Methods.infinity();
    }

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif 
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_nonstationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif

    int tid = 0;
    
    uint32_t xo = accu_segment_cols[tid];
    auto* x_data = (Fractional_Type*) X[xo];
    auto* xi_data = (Integer_Type*) XI[xo];
    auto* xv_data = (Fractional_Type*) XV[xo];
    Integer_Type x_nitems = nnz_col_sizes_loc[xo];
    const auto* JC = colgrp_nnz_cols[tid];
    Integer_Type JC_nitems = nnz_col_sizes_loc[xo];
    auto* v_data = (Vertex_State*) V[tid];
    const auto* c_data = (char*) C[tid];
    Integer_Type i = 0;
    Integer_Type l = 0;
    for(Integer_Type j = 0; j < JC_nitems; j++) {
        i = JC[j];
        Vertex_State& state = v_data[i];
        if(c_data[i]) {
            x_data[j] = Vertex_Methods.messenger(state); 
            xv_data[l] = x_data[j];
            xi_data[l] = j;
            l++;
        }
        else
            x_data[j] = Vertex_Methods.infinity();
    }

    if(activity_filtering) {
        msgs_activity_statuses[xo] = l;
        int nitems = msgs_activity_statuses[xo];
        // 0 all, 1 nothing, else nitems
        double ratio = (double) nitems/x_nitems;
        if(ratio <= activity_filtering_ratio)
            nitems++;
        else
            nitems = 0;
        msgs_activity_statuses[xo] = nitems;
        activity_statuses[owned_segments[tid]] = msgs_activity_statuses[xo];
        
        for(int32_t r = 0; r < Env::nranks; r++) {    
            if(r != Env::rank) {
                int32_t j = tid + (r * num_owned_segments);
                int32_t m = owned_segments_all[j];
                MPI_Sendrecv(&activity_statuses[owned_segments[tid]], 1, TYPE_INT, r, tid, 
                             &activity_statuses[m], 1, TYPE_INT, r, tid, Env::MPI_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Request request;
    int32_t leader_cg, col_group;    
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
    //for(int32_t i: colgrp_owner_thread_segments[tid]) {
        col_group = local_col_segments[i];
        leader_cg = leader_ranks_cg[col_group];  
        
        int nitems = 0;
        if(Env::rank_cg == leader_cg)
            nitems = msgs_activity_statuses[i];
        MPI_Ibcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicators[tid], &request);
        MPI_Wait(&request, MPI_STATUSES_IGNORE);
        if(Env::rank_cg != leader_cg)
            msgs_activity_statuses[i] = nitems;
        if(activity_filtering and nitems) {
            auto* xij_data = (Integer_Type*) XI[i];
            auto* xvj_data = (Fractional_Type*) XV[i];
            if(nitems > 1) {
                MPI_Ibcast(xij_data, nitems - 1, TYPE_INT, leader_cg, colgrps_communicators[tid], &request);
                out_requests_t[tid].push_back(request);
                MPI_Ibcast(xvj_data, nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicators[tid], &request);
                out_requests_t[tid].push_back(request);
            }
        }
        else {
            auto* xj_data = (Fractional_Type*) X[i];
            Integer_Type xj_nitems = nnz_col_sizes_loc[i];
            MPI_Ibcast(xj_data, xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicators[tid], &request);
            out_requests_t[tid].push_back(request);
        }
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();     

    #ifdef TIMING 
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Bcast", elapsed_time);
    bcast_time.push_back(elapsed_time);
    #endif       
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_nonstationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    //for(uint32_t t: local_tiles_row_order_t[tid]) {
    for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        tile_th = tile.nth;
        pair_idx = pair.row;
        yi = tile.ith;
        auto* y_data = (Fractional_Type*) Y[yi];
        Integer_Type y_nitems = nnz_row_sizes_loc[yi];
        const auto* x_data = (Fractional_Type*) X[xi];
        const auto* xv_data = (Fractional_Type*) XV[xi];
        const auto* xi_data = (Integer_Type*) XI[xi];
        auto* t_data = (char*) T[yi];
        
        if(compression_type == _TCSC_) {
            #pragma omp parallel
            {   
                int tid = omp_get_thread_num();
                int ret = Env::set_thread_affinity(tid);
                const uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                if(nnz) {
                    #ifdef HAS_WEIGHT
                    const Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                    #endif
                    const Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                    const Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                    const Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
                    
                    if(activity_filtering and activity_statuses[tile.cg]) {
                        Integer_Type s_nitems = msgs_activity_statuses[tile.jth] - 1;
                        Integer_Type j = 0;
                        for(Integer_Type k = 0; k < s_nitems; k++) {
                            j = xi_data[k];
                            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                #ifdef HAS_WEIGHT
                                Vertex_Methods.combiner(y_data[IA[i]], xv_data[k], A[i]);
                                #else
                                Vertex_Methods.combiner(y_data[IA[i]], xv_data[k]);
                                #endif
                                t_data[IA[i]] = 1;
                            }
                        }
                    }
                    else {
                        for(uint32_t j = 0; j < ncols; j++) {
                            if(x_data[j] != Vertex_Methods.infinity()) {
                                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                    #ifdef HAS_WEIGHT
                                    Vertex_Methods.combiner(y_data[IA[i]], x_data[j], A[i]);
                                    #else
                                    Vertex_Methods.combiner(y_data[IA[i]], x_data[j]);
                                    #endif
                                    t_data[IA[i]] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        else {  
            #pragma omp parallel
            {   
                int tid = omp_get_thread_num();
                int ret = Env::set_thread_affinity(tid);        
                const uint64_t nnz = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                if(nnz) {
                    #ifdef HAS_WEIGHT
                    const Integer_Type* A = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                    #endif
                    const Integer_Type* IA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                    const Integer_Type* JA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                    const Integer_Type ncols = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;        
                    Integer_Type l;
                    if(activity_filtering and activity_statuses[tile.cg]) {
                        if(iteration == 0) {               
                            Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_REG_R_SNK_C;
                            if(NC_REG_R_SNK_C) {
                                Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_REG_R_SNK_C;
                                Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_REG_R_SNK_C; j++, k = k + 2) {
                                    l = JC_REG_R_SNK_C[j];
                                    for(uint32_t i = JA_REG_R_SNK_C[k]; i < JA_REG_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                        t_data[IA[i]] = 1;
                                    }
                                }                    
                            }
                        }
                        if(not converged) {
                            //if(ncols) {
                                //Integer_Type* JC_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_NNZ_C;
                                Integer_Type* JA_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_NNZ_C;
                                Integer_Type s_nitems = msgs_activity_statuses[tile.jth] - 1;
                                Integer_Type j = 0;
                                for(Integer_Type k = 0; k < s_nitems; k++) {
                                    j = xi_data[k] * 2;
                                    //for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                    for(uint32_t i = JA_REG_R_NNZ_C[j]; i < JA_REG_R_NNZ_C[j + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], xv_data[k], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], xv_data[k]);
                                        #endif
                                        t_data[IA[i]] = 1;
                                    }
                                }                   
                            //}
                        }
                        else {
                            Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_REG_C;
                            if(NC_SRC_R_REG_C) {
                                Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_REG_C;
                                Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_REG_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_REG_C; j++, k = k + 2) {
                                    l = JC_SRC_R_REG_C[j];
                                    for(uint32_t i = JA_SRC_R_REG_C[k]; i < JA_SRC_R_REG_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                            Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_SNK_C;
                            if(NC_SRC_R_SNK_C) {
                                Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_SNK_C;
                                Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_SNK_C; j++, k = k + 2) {
                                    l = JC_SRC_R_SNK_C[j];
                                    for(uint32_t i = JA_SRC_R_SNK_C[k]; i < JA_SRC_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                        }
                    }
                    else { 
                        if(iteration == 0) {               
                            Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_REG_R_SNK_C;
                            if(NC_REG_R_SNK_C) {
                                Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_REG_R_SNK_C;
                                Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_REG_R_SNK_C; j++, k = k + 2) {
                                    l = JC_REG_R_SNK_C[j];
                                    for(uint32_t i = JA_REG_R_SNK_C[k]; i < JA_REG_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                        t_data[IA[i]] = 1;
                                    }
                                }                    
                            }
                        }
                        if(not converged) {
                            //Integer_Type NC_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_NNZ_C;
                            //if(NC_REG_R_NNZ_C) {
                                //Integer_Type* JC_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_NNZ_C;
                                Integer_Type* JA_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_REG_R_NNZ_C;
                                for(uint32_t j = 0, k = 0; j < ncols; j++, k = k + 2) {
                                    //l = j;
                                    if(x_data[j] != Vertex_Methods.infinity()) {
                                        for(uint32_t i = JA_REG_R_NNZ_C[k]; i < JA_REG_R_NNZ_C[k + 1]; i++) {
                                            #ifdef HAS_WEIGHT
                                            Vertex_Methods.combiner(y_data[IA[i]], x_data[j], A[i]);
                                            #else
                                            Vertex_Methods.combiner(y_data[IA[i]], x_data[j]);
                                            #endif
                                            t_data[IA[i]] = 1;
                                        }
                                    }
                                }                    
                            //}
                        }
                        else {
                            Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_REG_C;
                            if(NC_SRC_R_REG_C) {
                                Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_REG_C;
                                Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_REG_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_REG_C; j++, k = k + 2) {
                                    l = JC_SRC_R_REG_C[j];
                                    for(uint32_t i = JA_SRC_R_REG_C[k]; i < JA_SRC_R_REG_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                            Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->NC_SRC_R_SNK_C;
                            if(NC_SRC_R_REG_C) {
                                Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JC_SRC_R_SNK_C;
                                Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA_SRC_R_SNK_C;
                                for(uint32_t j = 0, k = 0; j < NC_SRC_R_SNK_C; j++, k = k + 2) {
                                    l = JC_SRC_R_SNK_C[j];
                                    for(uint32_t i = JA_SRC_R_SNK_C[k]; i < JA_SRC_R_SNK_C[k + 1]; i++) {
                                        #ifdef HAS_WEIGHT
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l], A[i]);
                                        #else
                                        Vertex_Methods.combiner(y_data[IA[i]], x_data[l]);
                                        #endif
                                    }
                                }                    
                            }
                        }
                    }
                }
            }
        }
                
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            int tid = 0;
            leader = tile.leader_rank_rg_rg;
            my_rank = Env::rank_rg;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
                    follower = follower_rowgrp_ranks_rg[j];
                    if(activity_filtering and activity_statuses[tile.rg]) {
                        // 0 all / 1 nothing / else nitems 
                        int nitems = 0;
                        MPI_Status status;
                        MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, rowgrps_communicators[yi], &status);
                        accus_activity_statuses[tid][j] = nitems;
                        if(accus_activity_statuses[tid][j] > 1) {
                            auto* yij_data = (Integer_Type*) YIt[tid][j];
                            auto* yvj_data = (Fractional_Type*) YVt[tid][j];
                            MPI_Irecv(yij_data, accus_activity_statuses[tid][j] - 1, TYPE_INT, follower, pair_idx, rowgrps_communicators[yi], &request);
                            in_requests_t[tid].push_back(request);
                            MPI_Irecv(yvj_data, accus_activity_statuses[tid][j] - 1, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[yi], &request);
                            in_requests_t[tid].push_back(request);
                        }
                    }
                    else {                                
                        auto* yj_data = (Fractional_Type*) Yt[tid][j];
                        Integer_Type yj_nitems = nnz_row_sizes_all[owned_segments[tid]];
                        MPI_Irecv(yj_data, yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[yi], &request);
                        in_requests_t[tid].push_back(request);
                    }       
                }   
            }
            else {
                auto* yi_data = (Integer_Type*) YI[yi];
                auto* yv_data = (Fractional_Type*) YV[yi];
                int nitems = 0;
                if(activity_filtering and activity_statuses[tile.rg]) {
                    char* t_data = (char*) T[yi];
                    Integer_Type j = 0;
                    for(uint32_t i = 0; i < y_nitems; i++) {
                        if(t_data[i]) {
                            yi_data[j] = i;
                            yv_data[j] = y_data[i];
                            j++;
                        }
                    }
                    nitems = j;
                    nitems++;
                    MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, rowgrps_communicators[yi]);
                    if(nitems > 1) {
                        MPI_Isend(yi_data, nitems - 1, TYPE_INT, leader, pair_idx, rowgrps_communicators[yi], &request);
                        out_requests_t[tid].push_back(request);
                        MPI_Isend(yv_data, nitems - 1, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[yi], &request);
                        out_requests_t[tid].push_back(request);
                    }
                }
                else {
                    MPI_Isend(y_data, y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[yi], &request);
                    out_requests_t[tid].push_back(request);
                }
            }
            xi = 0;
        }

    }
    int tid = 0;
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();
    
    yi = accu_segment_rows[tid];
    auto* y_data = (Fractional_Type*) Y[yi];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        if(activity_filtering and accus_activity_statuses[tid][j]) {
            if(accus_activity_statuses[tid][j] > 1) {
                auto* yij_data = (Integer_Type*) YIt[tid][j];
                auto* yvj_data = (Fractional_Type*) YVt[tid][j];
                for(uint32_t i = 0; i < accus_activity_statuses[tid][j] - 1; i++) {
                    Integer_Type k = yij_data[i];
                    Vertex_Methods.combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else {
            auto* yj_data = (Fractional_Type*) Yt[tid][j];
            Integer_Type yj_nitems = nnz_row_sizes_all[owned_segments[tid]];
            for(uint32_t i = 0; i < yj_nitems; i++)
                Vertex_Methods.combiner(y_data[i], yj_data[i]);
        }
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    
    #ifdef TIMING   
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Combine", elapsed_time);
    combine_time.push_back(elapsed_time);
    #endif  
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_nonstationary_2dgp() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
    int tid = 0;
    
    uint32_t yi = accu_segment_rows[tid];
    auto* y_data = (Fractional_Type*) Y[yi];
    auto* i_data = (char*) I[yi];
    auto* iv_data = (Integer_Type*) IV[yi];
    auto* reg_nnz_data = (Integer_Type*) rowgrp_nnz_rows[tid];
    Integer_Type reg_nnz_nitems = nnz_row_sizes_loc[yi];
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    Integer_Type v_nitems = tile_height;
    if(computation_filtering) {
        auto* regular_rows = (Integer_Type*) rowgrp_regular_rows[tid];
        Integer_Type reg_nitems = rowgrp_regular_rows_blks[tid].nitems; 
        auto* source_rows = (Integer_Type*) rowgrp_source_rows[tid];
        Integer_Type src_nitems = rowgrp_source_rows_blks[tid].nitems; 
        if(apply_depends_on_iter) {
            if(iteration == 0) {
                for(uint32_t i = 0; i < v_nitems; i++) {
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j], iteration);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
            else {
                if(not converged) {
                    for(Integer_Type r = 0; r < reg_nitems; r++) {
                        Integer_Type i = regular_rows[r];
                        Vertex_State& state = v_data[i];
                        Integer_Type j = iv_data[i];    
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j], iteration);
                    }
                }
                else {
                    for(Integer_Type r = 0; r < src_nitems; r++) {
                        Integer_Type i = source_rows[r];
                        Vertex_State& state = v_data[i];
                        Integer_Type j = iv_data[i];    
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j], iteration);
                    }
                }
            }
        }
        else {
            if(iteration == 0) {            
                for(uint32_t i = 0; i < v_nitems; i++) {
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
            else {
                if(not converged) {
                    for(Integer_Type r = 0; r < reg_nitems; r++) {
                        Integer_Type i = regular_rows[r];
                        Vertex_State& state = v_data[i];
                        Integer_Type j = iv_data[i];    
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
                    }
                }
                else {
                    for(Integer_Type r = 0; r < src_nitems; r++) {
                        Integer_Type i = source_rows[r];
                        Vertex_State& state = v_data[i];
                        Integer_Type j = iv_data[i];    
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
                    }
                }
            }
        }
    }
    else {
        if(apply_depends_on_iter) {
            if(iteration == 0) {
                for(uint32_t i = 0; i < v_nitems; i++) {
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i]; 
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j], iteration);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
            else {
                for(Integer_Type r = 0; r < reg_nnz_nitems; r++) {
                    Integer_Type i = reg_nnz_data[r];
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[j], iteration);
                }
            }
        }
        else {
            if(iteration == 0) {
                for(Integer_Type i = 0; i < v_nitems; i++) {
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i];  
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
            else {
                for(Integer_Type r = 0; r < reg_nnz_nitems; r++) {
                    Integer_Type i = reg_nnz_data[r];
                    Vertex_State& state = v_data[i];
                    Integer_Type j = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[j]);
                }
            }
        }
    }
    #ifdef TIMING  
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Apply", elapsed_time);
    apply_time.push_back(elapsed_time);
    #endif   
}



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::has_converged_2dgp() {    
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif    
    int tid = 0;
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;    
    if(check_for_convergence) {
        uint32_t yi  = accu_segment_rows[tid];
        auto* c_data = (char*) C[tid];
        auto* iv_data = (Integer_Type*) IV[yi];        
        convergence_vec[tid] = 0;   
        if(computation_filtering) {
            auto* regular_rows = (Integer_Type*) rowgrp_regular_rows[tid];
            Integer_Type reg_nitems = rowgrp_regular_rows_blks[tid].nitems;
            for(Integer_Type r = 0; r < reg_nitems; r++) {
                Integer_Type i = regular_rows[r];
                if(not c_data[i]) {
                    c_sum_local++;
                }
            }
            if(c_sum_local == reg_nitems)
                convergence_vec[tid] = 1;
        }
        else {
            for(uint32_t i = 0; i < tile_height; i++) {
                if(not c_data[i]) {
                    c_sum_local++;
                }
            }
            if(c_sum_local == tile_height)
                convergence_vec[tid] = 1;
        }
    }
    

    iteration++;
    converged = false;
    if(check_for_convergence) {
        if(convergence_vec[tid] == 1)
            c_sum_local = 1;
        else 
            c_sum_local = 0;
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        
        if(c_sum_gloabl == (uint64_t) Env::nranks)
            converged = true;
    }
    else if(iteration >= num_iterations)
            converged = true;
    
        
    if(converged){
        if(computation_filtering) { 
            if(stationary) {
                combine_2d_stationary_2dgp();
                apply_stationary_2dgp();
            }
            else {
                combine_2d_nonstationary_2dgp();
                apply_nonstationary_2dgp();
            }
        }
    }
    
    if((stationary) or ((not stationary) and ((not gather_depends_on_apply) and (not apply_depends_on_iter)))) {
        for(uint32_t j = 0; j < rank_nrowgrps; j++) {
            auto* y_data = (Fractional_Type*) Y[j];
            uint64_t nbytes = Y_blks[j].nbytes;
            memset(y_data, 0, nbytes);  
        }
        
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            auto* yt_data = (Fractional_Type*) Yt[tid][j];
            uint64_t nbytes = Yt_blks[tid][j].nbytes;
            memset(yt_data, 0, nbytes); 
        }
        
    }

    if(not stationary) {
        for(uint32_t j = 0; j < rank_nrowgrps; j++) {
            auto* t_data = (char*) T[j];
            uint64_t nbytes = T_blks[j].nbytes;
            memset(t_data, 0, nbytes); 
        }
        std::fill(accus_activity_statuses[tid].begin(), accus_activity_statuses[tid].end(), 0);
        for(uint32_t j = 0; j < rank_ncolgrps; j++) {
            msgs_activity_statuses[j] = 0;    
        }
    }        
    
    #ifdef TIMING  
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Convergence", elapsed_time);
    convergence_time.push_back(elapsed_time);
    #endif   
    
    Env::print_num("Iteration", iteration);
    return(converged);   
}

#endif