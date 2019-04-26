/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include <type_traits>
#include <numeric>

#include "mpi/types.hpp" 
#include "ds/numa_vector.hpp"

struct State { State() {}; };


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
class Vertex_Program
{
    public:
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph,
                        bool stationary_ = false, bool gather_depends_on_apply_ = false, 
                        bool apply_depends_on_iter_ = false);
        ~Vertex_Program();
        Vertex_Methods_Impl Vertex_Methods;
        Vertex_Program(Vertex_Methods_Impl const &VMs) : Vertex_Methods(VMs) { };
        void set_root(Integer_Type root_) { 
            Vertex_Methods.set_root(hasher->hash(root_));
        };
        
        void execute(Integer_Type num_iterations_ = 0);
        template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_, typename Vertex_Methods_Impl_>
        void initialize(const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_, Vertex_Methods_Impl_>& VProgram);
        void free();
        void checksum();
        void display(Integer_Type count = 31);
        
        Integer_Type num_iterations = 0;
        bool stationary = false;
        bool gather_depends_on_apply = false;
        bool apply_depends_on_iter = false;
        Integer_Type iteration = 0;
        Vertex_State** V = nullptr;
    protected:
        bool already_initialized = false;
        bool check_for_convergence = false;
        bool converged = false;
        void init_stationary(int tid);
        void init_nonstationary(int tid);
        void init_vectors();
        void bcast_stationary(int tid);
        void bcast_nonstationary(int tid);
        void combine_2d_stationary(int tid);
        void combine_2d_nonstationary(int tid);
        void apply_stationary(int tid);
        void apply_nonstationary(int tid);

        void thread_function_stationary(int tid);
        void thread_function_nonstationary(int tid);

        bool has_converged(int tid);
        
        Integer_Type get_vid(Integer_Type index, int32_t segment);
        
        Tiling_type tiling_type;
        Integer_Type nrows, ncols;
        uint32_t nrowgrps, ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nrowgrps, rank_ncolgrps;
        Integer_Type tile_height, tile_width;
        std::vector<int32_t> local_col_segments;
        std::vector<int32_t> accu_segment_col_vec;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        std::vector<int32_t> local_row_segments;
        std::vector<int32_t> all_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_rg;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg_rg;
        std::vector<int32_t> follower_colgrp_ranks_cg;
        std::vector<int32_t> follower_colgrp_ranks;
        std::vector<int32_t> leader_ranks;
        std::vector<int32_t> leader_ranks_cg;
        std::vector<uint32_t> local_tiles_row_order;
        std::vector<uint32_t> local_tiles_col_order;

        std::vector<int32_t> owned_segments;
        std::vector<int32_t> owned_segments_all;
        int32_t num_owned_segments;
        std::vector<std::vector<uint32_t>> local_tiles_row_order_t;
        std::vector<std::vector<uint32_t>> local_tiles_col_order_t;
        
        std::vector<int32_t> follower_rowgrp_ranks;
        std::vector<int32_t> follower_rowgrp_ranks_accu_seg;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        std::vector<MPI_Request> out_requests_;
        std::vector<MPI_Request> in_requests_;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
        
        std::vector<std::vector<MPI_Request>> out_requests_t;
        std::vector<std::vector<MPI_Request>> in_requests_t;
    

        
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        std::vector<int32_t> accu_segment_rows, accu_segment_cols;
        std::vector<int32_t> rowgrp_owner_thread, colgrp_owner_thread;
        std::vector<std::vector<int32_t>> rowgrp_owner_thread_segments, colgrp_owner_thread_segments;
        std::vector<int32_t> convergence_vec;
        
        Matrix<Weight, Integer_Type, Fractional_Type>* A; // Adjacency list        
        
        char** C = nullptr;
        Fractional_Type** X = nullptr;
        Fractional_Type** Y = nullptr;
        std::vector<Fractional_Type**> Yt;
        
        std::vector<struct blk<Integer_Type>> V_blks;
        std::vector<struct blk<Integer_Type>> C_blks;
        std::vector<struct blk<Integer_Type>> X_blks;
        std::vector<struct blk<Integer_Type>> Y_blks;        
        std::vector<std::vector<struct blk<Integer_Type>>> Yt_blks;
        
        
        /* Nonstationary */
        Integer_Type** XI;                  // X Indices (Nonstationary)
        Fractional_Type** XV;               // X Values  (Nonstationary)
        Integer_Type** YI;                  // Y Indices (Nonstationary)
        Fractional_Type** YV;               // Y Values (Nonstationary)
        std::vector<Integer_Type**> YIt;    // Partial Y Indices (Nonstationary)
        std::vector<Fractional_Type**> YVt; // Partial Y Values (Nonstationary)
        char** T;                           // Accumulators activity vectors
        
        std::vector<struct blk<Integer_Type>> XI_blks;
        std::vector<struct blk<Integer_Type>> XV_blks;
        std::vector<struct blk<Integer_Type>> YI_blks;
        std::vector<struct blk<Integer_Type>> YV_blks;
        std::vector<std::vector<struct blk<Integer_Type>>> YIt_blks;
        std::vector<std::vector<struct blk<Integer_Type>>> YVt_blks;
        std::vector<struct blk<Integer_Type>> T_blks;
        
        std::vector<Integer_Type> msgs_activity_statuses;
        std::vector<std::vector<Integer_Type>> accus_activity_statuses;
        std::vector<Integer_Type> activity_statuses;
        std::vector<Integer_Type> active_vertices;
        
        char** I = nullptr;
        Integer_Type** IV = nullptr;
        char** J = nullptr;
        Integer_Type** JV = nullptr;
        Integer_Type** rowgrp_nnz_rows = nullptr;
        Integer_Type** colgrp_nnz_cols = nullptr;
        
        std::vector<Integer_Type> nnz_row_sizes_loc;
        std::vector<Integer_Type> nnz_col_sizes_loc;
        std::vector<Integer_Type> nnz_row_sizes_all;
        std::vector<Integer_Type> nnz_col_sizes_all;
        
        MPI_Datatype TYPE_DOUBLE;
        MPI_Datatype TYPE_INT;
        MPI_Datatype TYPE_CHAR;
        
        bool directed;
        bool transpose;
        double activity_filtering_ratio = 0.6;
        bool activity_filtering = true;
        
        bool computation_filering = false;
        Integer_Type** rowgrp_regular_rows;
        Integer_Type** rowgrp_source_rows;
        Integer_Type** colgrp_sink_cols;
        std::vector<struct blk<Integer_Type>> rowgrp_regular_rows_blks;
        std::vector<struct blk<Integer_Type>> rowgrp_source_rows_blks;
        std::vector<struct blk<Integer_Type>> colgrp_sink_cols_blks;
        
        std::vector<MPI_Comm> rowgrps_communicators;
        std::vector<MPI_Comm> colgrps_communicators;
        
        pthread_barrier_t p_barrier;
        ReversibleHasher *hasher;
        Compression_type compression_type;
        #ifdef TIMING
        void times();
        void stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev);
        std::vector<double> init_time;
        std::vector<double> bcast_time;
        std::vector<double> combine_time;
        std::vector<double> apply_time;
        std::vector<double> convergence_time;
        std::vector<double> execute_time;
        #endif
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph, bool stationary_, 
         bool gather_depends_on_apply_, bool apply_depends_on_iter_) {

    A = Graph.A;
    directed = A->directed;
    transpose = A->transpose;
    stationary = stationary_;
    gather_depends_on_apply = gather_depends_on_apply_;
    apply_depends_on_iter = apply_depends_on_iter_;
    tiling_type = A->tiling->tiling_type;
    leader_ranks = A->leader_ranks;
    owned_segments = Graph.A->owned_segments;
    owned_segments_all = Graph.A->owned_segments_all;
    rowgrp_owner_thread_segments = Graph.A->rowgrp_owner_thread_segments;
    colgrp_owner_thread_segments = Graph.A->colgrp_owner_thread_segments;
    rowgrp_owner_thread = Graph.A->rowgrp_owner_thread;
    colgrp_owner_thread = Graph.A->colgrp_owner_thread;
    
    compression_type = A->compression_type;
    if(compression_type == _TCSC_CF_ and directed) {
        computation_filering = true;
        rowgrp_regular_rows = Graph.A->rowgrp_regular_rows;
        rowgrp_source_rows = Graph.A->rowgrp_source_rows;
        colgrp_sink_cols = Graph.A->colgrp_sink_cols;
        rowgrp_regular_rows_blks = Graph.A->rowgrp_regular_rows_blks;
        rowgrp_source_rows_blks = Graph.A->rowgrp_source_rows_blks;
        colgrp_sink_cols_blks = Graph.A->colgrp_sink_cols_blks;
    }
    
    hasher = A->hasher;
    convergence_vec.resize(Env::nthreads);
    pthread_barrier_init(&p_barrier, NULL, Env::nthreads);

    nrows = A->nrows;
    ncols = A->ncols;
    nrowgrps = A->nrowgrps;
    ncolgrps = A->ncolgrps;
    rowgrp_nranks = A->tiling->rowgrp_nranks;
    colgrp_nranks = A->tiling->colgrp_nranks;
    rank_nrowgrps = A->tiling->rank_nrowgrps;
    rank_ncolgrps = A->tiling->rank_ncolgrps;
    tile_height = A->tile_height;
    tile_width = A->tile_width;
    local_row_segments = A->local_row_segments;
    local_col_segments = A->local_col_segments;
    all_rowgrp_ranks_accu_seg = A->all_rowgrp_ranks_accu_seg;
    follower_rowgrp_ranks_rg = A->follower_rowgrp_ranks_rg;
    follower_rowgrp_ranks_accu_seg_rg = A->follower_rowgrp_ranks_accu_seg_rg;
    leader_ranks_cg = A->leader_ranks_cg;
    follower_colgrp_ranks_cg = A->follower_colgrp_ranks_cg;
    follower_colgrp_ranks = A->follower_colgrp_ranks;
    local_tiles_row_order = A->local_tiles_row_order;
    local_tiles_col_order = A->local_tiles_col_order;
    follower_rowgrp_ranks = A->follower_rowgrp_ranks;
    follower_rowgrp_ranks_accu_seg = A->follower_rowgrp_ranks_accu_seg;
    all_rowgrp_ranks = A->all_rowgrp_ranks;
    nnz_row_sizes_loc = A->nnz_row_sizes_loc;
    nnz_col_sizes_loc = A->nnz_col_sizes_loc;
    nnz_row_sizes_all = A->nnz_row_sizes_all;
    nnz_col_sizes_all = A->nnz_col_sizes_all;
        
    I = Graph.A->I;
    IV = Graph.A->IV;
    J = Graph.A->J;
    JV = Graph.A->JV;
    rowgrp_nnz_rows = Graph.A->rowgrp_nnz_rows;
    colgrp_nnz_cols = Graph.A->colgrp_nnz_cols;
                
    out_requests_t.resize(Env::nthreads);
    in_requests_t.resize(Env::nthreads);

    num_owned_segments = Graph.A->num_owned_segments;
    local_tiles_row_order_t = Graph.A->local_tiles_row_order_t;
    local_tiles_col_order_t = Graph.A->local_tiles_col_order_t;
        
    accu_segment_rows = Graph.A->accu_segment_rows;
    accu_segment_cols = Graph.A->accu_segment_cols;
    
    rowgrps_communicators = Env::rowgrps_comms;
    colgrps_communicators = Env::colgrps_comms;
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::execute(Integer_Type num_iterations_) {
    
    num_iterations = num_iterations_;
    if(!num_iterations)
        check_for_convergence = true; 
    
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    
    if(stationary) {
        std::vector<std::thread> threads;
        for(int i = 0; i < Env::nthreads; i++) {
            threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_stationary, this, i));
        }
        
        for(std::thread& th: threads) {
            th.join();
        }
    }
    else {
        std::vector<std::thread> threads;
        for(int i = 0; i < Env::nthreads; i++) {
            threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_nonstationary, this, i));
        }
        
        for(std::thread& th: threads) {
            th.join();
        }
    }
    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Execute", elapsed_time);
    #ifdef TIMING
    execute_time.push_back(elapsed_time);
    times();
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_vectors() {
    std::vector<Integer_Type> c_v_sizes(num_owned_segments, tile_height);
    std::vector<int32_t> thread_sockets(num_owned_segments);
    for(int i = 0; i < Env::nthreads; i++) {
        thread_sockets[i] = Env::socket_of_thread(i);
    }
    V_blks.resize(num_owned_segments);
    C_blks.resize(num_owned_segments);
    for(int32_t j = 0; j < num_owned_segments; j++) {  
        auto& v_blk = V_blks[j];
        v_blk.nitems = c_v_sizes[j];
        v_blk.socket_id = thread_sockets[j];
        auto& c_blk = C_blks[j];
        c_blk.nitems = c_v_sizes[j];
        c_blk.socket_id = thread_sockets[j];
    }
    allocate_numa_vector<Integer_Type, Vertex_State>(&V, V_blks);
    allocate_numa_vector<Integer_Type, char>(&C, C_blks);
    
    // Initialize messages
    std::vector<Integer_Type> x_sizes = nnz_col_sizes_loc;
    std::vector<int32_t> x_thread_sockets(rank_ncolgrps);    
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        x_thread_sockets[i] = Env::socket_of_thread(colgrp_owner_thread[i]);
    }

    X_blks.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        auto& blk = X_blks[i];
        blk.nitems = x_sizes[i];
        blk.socket_id = x_thread_sockets[i];
    }
    allocate_numa_vector<Integer_Type, Fractional_Type>(&X, X_blks);

    // Initialiaze accumulators
    std::vector<Integer_Type> y_sizes = nnz_row_sizes_loc;
    std::vector<int32_t> y_thread_sockets(rank_nrowgrps);    
    for(uint32_t i = 0; i < rank_nrowgrps; i++) {
        y_thread_sockets[i] = Env::socket_of_thread(rowgrp_owner_thread[i]);
    }
    Y_blks.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++) {
        auto& blk = Y_blks[i];
        blk.nitems = y_sizes[i];
        blk.socket_id = y_thread_sockets[i];
    }
    allocate_numa_vector<Integer_Type, Fractional_Type>(&Y, Y_blks);
    
    // Partial accumulators
    Yt.resize(num_owned_segments);
    Yt_blks.resize(num_owned_segments);
    for(int32_t i = 0; i < num_owned_segments; i++) {
        std::vector<Integer_Type> row_size((rowgrp_nranks - 1), nnz_row_sizes_all[owned_segments[i]]);
        std::vector<int32_t> row_socket((rowgrp_nranks - 1), thread_sockets[i]);
        Yt_blks[i].resize(rowgrp_nranks - 1);
        for(uint32_t j = 0; j < (rowgrp_nranks - 1); j++) {
            auto& blk = Yt_blks[i][j];
            blk.nitems = row_size[j];
            blk.socket_id = row_socket[j];
        }
        allocate_numa_vector<Integer_Type, Fractional_Type>(&Yt[i], Yt_blks[i]);
    }
    
    // Stationary vectors
    if(not stationary) {
        XI_blks.resize(rank_ncolgrps);
        XV_blks.resize(rank_ncolgrps);
        for(uint32_t i = 0; i < rank_ncolgrps; i++) {
            auto& xi_blk = XI_blks[i];
            xi_blk.nitems = x_sizes[i];
            xi_blk.socket_id = x_thread_sockets[i];
            auto& xv_blk = XV_blks[i];
            xv_blk.nitems = x_sizes[i];
            xv_blk.socket_id = x_thread_sockets[i];
        }
        allocate_numa_vector<Integer_Type, Integer_Type>(&XI, XI_blks);
        allocate_numa_vector<Integer_Type, Fractional_Type>(&XV, XV_blks);
        
        YI_blks.resize(rank_nrowgrps);
        YV_blks.resize(rank_nrowgrps);
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            auto& yi_blk = YI_blks[i];
            yi_blk.nitems = y_sizes[i];
            yi_blk.socket_id = y_thread_sockets[i];
            auto& yv_blk = YV_blks[i];
            yv_blk.nitems = y_sizes[i];
            yv_blk.socket_id = y_thread_sockets[i];
        }
        allocate_numa_vector<Integer_Type, Integer_Type>(&YI, YI_blks);
        allocate_numa_vector<Integer_Type, Fractional_Type>(&YV, YV_blks);
        
        YIt.resize(num_owned_segments);
        YVt.resize(num_owned_segments);
        YIt_blks.resize(num_owned_segments);
        YVt_blks.resize(num_owned_segments);
        for(int32_t i = 0; i < num_owned_segments; i++) {
            std::vector<Integer_Type> row_size((rowgrp_nranks - 1), nnz_row_sizes_all[owned_segments[i]]);
            std::vector<int32_t> row_socket((rowgrp_nranks - 1), thread_sockets[i]);
            YIt_blks[i].resize(rowgrp_nranks - 1);
            YVt_blks[i].resize(rowgrp_nranks - 1);
            for(uint32_t j = 0; j < (rowgrp_nranks - 1); j++) {
                auto& yit_blk = YIt_blks[i][j];
                yit_blk.nitems = row_size[j];
                yit_blk.socket_id = row_socket[j];
                auto& yvt_blk = YVt_blks[i][j];
                yvt_blk.nitems = row_size[j];
                yvt_blk.socket_id = row_socket[j];
            }
            allocate_numa_vector<Integer_Type, Integer_Type>(&YIt[i], YIt_blks[i]);
            allocate_numa_vector<Integer_Type, Fractional_Type>(&YVt[i], YVt_blks[i]);
        }
        
        T_blks.resize(rank_nrowgrps);
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            auto& blk = T_blks[i];
            blk.nitems = y_sizes[i];
            blk.socket_id = y_thread_sockets[i];
        }
        allocate_numa_vector<Integer_Type, char>(&T, T_blks);
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_, typename Vertex_Methods_Impl_>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::initialize(
    const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_, Vertex_Methods_Impl_>& VProgram) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
       
    if(stationary) {   
        init_vectors();
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            uint32_t yi = accu_segment_rows[tid];
            auto* i_data = (char*) I[yi];
            auto* v_data = (Vertex_State*) V[tid];
            auto* c_data = (char*) C[tid];
            for(uint32_t i = 0; i < tile_height; i++) {
                Vertex_State& state = v_data[i]; 
                if(i_data[i]) {
                    c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[tid]), state, (const State&) VProgram.V[tid][i]);
                }
            }
        }
    }
    else {
        fprintf(stderr, "ERROR(rank=%d): Not implemented\n", Env::rank);
        Env::exit(0);
    }
    already_initialized = true;

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif

}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_stationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif
    
    if(tid == 0) {
        init_vectors();
    }
    
    pthread_barrier_wait(&p_barrier);
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    for(uint32_t i = 0; i < tile_height; i++) {
        Vertex_State& state = v_data[i]; 
        c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[tid]), state);
    }
    pthread_barrier_wait(&p_barrier);
    
    #ifdef TIMING
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Init", elapsed_time);
        init_time.push_back(elapsed_time);
    }
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_stationary(int tid) {      
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif
    
    uint32_t xo = accu_segment_cols[tid];       
    auto* x_data = (Fractional_Type*) X[xo];
    const auto* JC = (Integer_Type*) colgrp_nnz_cols[tid];
    Integer_Type JC_nitems = nnz_col_sizes_loc[xo];
    auto* v_data = (Vertex_State*) V[tid];
    for(uint32_t j = 0; j < JC_nitems; j++) {
        Vertex_State& state = v_data[JC[j]];
        x_data[j] = Vertex_Methods.messenger(state);
    }
    pthread_barrier_wait(&p_barrier);

    MPI_Request request;
    int32_t leader, col_group;    
    for(int32_t i: colgrp_owner_thread_segments[tid]) {
        col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];  
        auto* xj_data = (Fractional_Type*) X[i];
        Integer_Type xj_nitems = nnz_col_sizes_loc[i];
        MPI_Ibcast(xj_data, xj_nitems, TYPE_DOUBLE, leader, colgrps_communicators[tid], &request);
        out_requests_t[tid].push_back(request);
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    pthread_barrier_wait(&p_barrier);
    
    #ifdef TIMING
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Bcast", elapsed_time);
        bcast_time.push_back(elapsed_time);
    }
    #endif
} 

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_stationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif

    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader;
    int32_t follower, my_rank, accu;
    bool communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        tile_th = tile.nth;
        pair_idx = pair.row;
        yi = tile.ith;
        auto* y_data = (Fractional_Type*) Y[yi];
        Integer_Type y_nitems = nnz_row_sizes_loc[yi];
        const auto* x_data = (Fractional_Type*) X[xi];
        if(compression_type == _TCSC_) {
            const uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
            if(nnz) {
                #ifdef HAS_WEIGHT
                const Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
                #endif
                const Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
                const Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
                const Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;  
                
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
            const uint64_t nnz = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
            if(nnz) {
                #ifdef HAS_WEIGHT
                const Integer_Type* A = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
                #endif
                const Integer_Type* IA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
                const Integer_Type* JA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
                const Integer_Type ncols = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
                /*
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
                */    
                    Integer_Type l;
                    if(iteration == 0) {               
                        Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_SNK_C;
                        if(NC_REG_R_SNK_C) {
                            Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_SNK_C;
                            Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_SNK_C;
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
                        Integer_Type NC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_REG_C;
                        if(NC_REG_R_REG_C) {
                            Integer_Type* JC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_REG_C;
                            Integer_Type* JA_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_REG_C;
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
                        Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_REG_C;
                        if(NC_SRC_R_REG_C) {
                            Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_REG_C;
                            Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_REG_C;
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
                        
                        Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_SNK_C;
                        if(NC_SRC_R_REG_C) {
                            Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_SNK_C;
                            Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_SNK_C;
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
                //}
            }
        }
        
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
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
    if(tid == 0) {    
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Combine", elapsed_time);
        combine_time.push_back(elapsed_time);
    }
    #endif    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_stationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif
    uint32_t yi  = accu_segment_rows[tid];
    const auto* i_data = (char*) I[yi];
    const auto* iv_data = (Integer_Type*) IV[yi];
    auto* y_data = (Fractional_Type*) Y[yi];
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];

    if(computation_filering) {
        /*
        Integer_Type i = 0;
        Integer_Type j = 0;
        uint32_t yi  = accu_segment_rows[tid];
        auto* i_data = (char*) I[yi];
        auto* iv_data = (Integer_Type*) IV[yi];
        auto* y_data = (Fractional_Type*) Y[yi];
        auto* v_data = (Vertex_State*) V[tid];
        auto* c_data = (char*) C[tid];
        */
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
        /*
        uint32_t yi  = accu_segment_rows[tid];
        const auto* y_data = (Fractional_Type*) Y[yi];
        auto* i_data = (char*) I[yi];
        auto* iv_data = (Integer_Type*) IV[yi];
        auto* v_data = (Vertex_State*) V[tid];
        auto* c_data = (char*) C[tid];
        */
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
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Apply", elapsed_time);
        apply_time.push_back(elapsed_time);
    }
    #endif
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_stationary(int tid) {
    int ret = Env::set_thread_affinity(tid);
    if(not already_initialized)
        init_stationary(tid);   
    
    do {
        bcast_stationary(tid);
        combine_2d_stationary(tid);
        apply_stationary(tid);
    } while(not has_converged(tid));     
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_nonstationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif    
    
    if(tid == 0) {  
        init_vectors();
        // Activity vectors
        activity_statuses.resize(ncolgrps);
        msgs_activity_statuses.resize(rank_ncolgrps);
        accus_activity_statuses.resize(Env::nthreads, std::vector<Integer_Type>(rowgrp_nranks - 1));
    }
    pthread_barrier_wait(&p_barrier);
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    for(uint32_t i = 0; i < tile_height; i++) {
        Vertex_State& state = v_data[i]; 
        c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[tid]), state);
    }
    for(int32_t i: rowgrp_owner_thread_segments[tid]) {
        auto* y_data = (Fractional_Type*) Y[i];
        Integer_Type y_nitems = nnz_row_sizes_loc[i];
        for(uint32_t j = 0; j < y_nitems; j++)
            y_data[j] = Vertex_Methods.infinity();
    }
    pthread_barrier_wait(&p_barrier);
    #ifdef TIMING
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Init", elapsed_time);
        init_time.push_back(elapsed_time);
    }
    #endif 
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_nonstationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif

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
    //pthread_barrier_wait(&p_barrier);
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
    pthread_barrier_wait(&p_barrier);

    MPI_Request request;
    int32_t leader_cg, col_group;    
    for(int32_t i: colgrp_owner_thread_segments[tid]) {
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
    pthread_barrier_wait(&p_barrier);
    
    #ifdef TIMING    
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Bcast", elapsed_time);
        bcast_time.push_back(elapsed_time);
    }
    #endif       
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_nonstationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif
    
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
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
            const uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
            if(nnz) {
                #ifdef HAS_WEIGHT
                const Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
                #endif
                const Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
                const Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
                const Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;  

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
        else {    
            const uint64_t nnz = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
            if(nnz) {
                #ifdef HAS_WEIGHT
                const Integer_Type* A = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
                #endif
                const Integer_Type* IA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
                const Integer_Type* JA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
                const Integer_Type ncols = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
                Integer_Type l;
                if(activity_filtering and activity_statuses[tile.cg]) {
                    if(iteration == 0) {               
                        Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_SNK_C;
                        if(NC_REG_R_SNK_C) {
                            Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_SNK_C;
                            Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_SNK_C;
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
                            Integer_Type* JA_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_NNZ_C;
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
                        Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_REG_C;
                        if(NC_SRC_R_REG_C) {
                            Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_REG_C;
                            Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_REG_C;
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
                        Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_SNK_C;
                        if(NC_SRC_R_SNK_C) {
                            Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_SNK_C;
                            Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_SNK_C;
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
                        Integer_Type NC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_SNK_C;
                        if(NC_REG_R_SNK_C) {
                            Integer_Type* JC_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_SNK_C;
                            Integer_Type* JA_REG_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_SNK_C;
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
                            Integer_Type* JA_REG_R_NNZ_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_NNZ_C;
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
                        Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_REG_C;
                        if(NC_SRC_R_REG_C) {
                            Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_REG_C;
                            Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_REG_C;
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
                        Integer_Type NC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_SNK_C;
                        if(NC_SRC_R_REG_C) {
                            Integer_Type* JC_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_SNK_C;
                            Integer_Type* JA_SRC_R_SNK_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_SNK_C;
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
                
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
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
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Combine", elapsed_time);
        combine_time.push_back(elapsed_time);
    }
    #endif  
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_nonstationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif

    uint32_t yi = accu_segment_rows[tid];
    auto* y_data = (Fractional_Type*) Y[yi];
    auto* i_data = (char*) I[yi];
    auto* iv_data = (Integer_Type*) IV[yi];
    auto* reg_nnz_data = (Integer_Type*) rowgrp_nnz_rows[tid];
    Integer_Type reg_nnz_nitems = nnz_row_sizes_loc[yi];
    auto* v_data = (Vertex_State*) V[tid];
    auto* c_data = (char*) C[tid];
    Integer_Type v_nitems = tile_height;
    if(computation_filering) {
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
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Apply", elapsed_time);
        apply_time.push_back(elapsed_time);
    }
    #endif   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_nonstationary(int tid) {
    int ret = Env::set_thread_affinity(tid);
    if(not already_initialized)
        init_nonstationary(tid);
    
    do {
        bcast_nonstationary(tid);
        combine_2d_nonstationary(tid);
        apply_nonstationary(tid);
    } while(not has_converged(tid));
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::has_converged(int tid) {    
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif    

    uint64_t c_sum_local = 0, c_sum_gloabl = 0;    
    if(check_for_convergence) {
        uint32_t yi  = accu_segment_rows[tid];
        auto* c_data = (char*) C[tid];
        auto* iv_data = (Integer_Type*) IV[yi];        
        convergence_vec[tid] = 0;   
        if(computation_filering) {
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
    
    pthread_barrier_wait(&p_barrier);
    if(tid == 0) {
        iteration++;
        converged = false;
        if(check_for_convergence) {
            if(std::accumulate(convergence_vec.begin(), convergence_vec.end(), 0) == Env::nthreads)
                c_sum_local = 1;
            else 
                c_sum_local = 0;
            MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
            
            if(c_sum_gloabl == (uint64_t) Env::nranks)
                converged = true;
        }
        else if(iteration >= num_iterations)
                converged = true;
    }
    pthread_barrier_wait(&p_barrier);
    
    if(converged){
        if(computation_filering) { 
            if(stationary) {
                combine_2d_stationary(tid);
                apply_stationary(tid);
            }
            else {
                combine_2d_nonstationary(tid);
                apply_nonstationary(tid);
            }
        }
    }
    if((stationary) or ((not stationary) and ((not gather_depends_on_apply) and (not apply_depends_on_iter)))) {
        
        for(int32_t j: rowgrp_owner_thread_segments[tid]) {
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
        for(int32_t j: rowgrp_owner_thread_segments[tid]) {
            auto* t_data = (char*) T[j];
            uint64_t nbytes = T_blks[j].nbytes;
            memset(t_data, 0, nbytes); 
        }
        std::fill(accus_activity_statuses[tid].begin(), accus_activity_statuses[tid].end(), 0);
        for(int32_t j: colgrp_owner_thread_segments[tid]) {
            msgs_activity_statuses[j] = 0;    
        }
    }        
    pthread_barrier_wait(&p_barrier);
    
    #ifdef TIMING    
    if(tid == 0) {
        t2 = Env::clock();
        elapsed_time = t2 - t1;
        Env::print_time("Convergence", elapsed_time);
        convergence_time.push_back(elapsed_time);
    }
    #endif   
    
    if(tid == 0) {
        Env::print_num("Iteration", iteration);
    }
    return(converged);   
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::checksum() {
    uint64_t v_sum_local = 0, v_sum_global = 0;
    for(int32_t k = 0; k < num_owned_segments; k++) {
        auto* v_data = (Vertex_State*) V[k];
        for(uint32_t i = 0; i < tile_height; i++) {
            Vertex_State &state = v_data[i];
            if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i, owned_segments[k]) < nrows))    
                    v_sum_local += state.get_state();
        }
    }
    
    MPI_Allreduce(&v_sum_local, &v_sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master) {
        std::cout << "INFO(rank=" << Env::rank << "): " << "Number of iterations: " << iteration << std::endl;
        std::cout << "INFO(rank=" << Env::rank << "): " << std::fixed << "Value checksum: " << v_sum_global << std::endl;
    }
    
    uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
    for(int32_t k = 0; k < num_owned_segments; k++) {
        auto* v_data = (Vertex_State*) V[k];
        for(uint32_t i = 0; i < tile_height; i++) {
            Vertex_State &state = v_data[i];
            if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i, owned_segments[k]) < nrows)) 
                v_sum_local_++;
        }
    }
    MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        std::cout << "INFO(rank=" << Env::rank << "): " << std::fixed << "Reachable vertices: " << v_sum_global_ << std::endl;
    Env::barrier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::display(Integer_Type count) {
    Integer_Type v_nitems = tile_height;
    count = (v_nitems < count) ? v_nitems : count;
    Env::barrier();
    
    uint32_t t = local_tiles_row_order[0];
    auto pair = A->tile_of_local_tile(t);
    auto& tile = A->tiles[pair.row][pair.col];
    if(!tile.kth) {
    //if(Env::is_master) {
        Triple<Weight, Integer_Type> pair, pair1;
        auto* v_data = (Vertex_State*) V[0];
        for(uint32_t i = 0; i < count; i++) {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segments[0], owned_segments[0]);
            Vertex_State& state = v_data[i];
            
            std::cout << std::fixed <<  "vertex[" << hasher->unhash(pair1.row) << "]:" << state.print_state() << std::endl;
            //std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
        }
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Integer_Type Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::get_vid(Integer_Type index, int32_t segment) {
    return(index + (segment * tile_height));
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::free() {
    deallocate_numa_vector<Integer_Type, Vertex_State>(&V, V_blks);
    deallocate_numa_vector<Integer_Type, char>(&C, C_blks);
    deallocate_numa_vector<Integer_Type, Fractional_Type>(&X, X_blks);
    deallocate_numa_vector<Integer_Type, Fractional_Type>(&Y, Y_blks);
    for(int i = 0; i < num_owned_segments; i++) {
        deallocate_numa_vector<Integer_Type, Fractional_Type>(&Yt[i], Yt_blks[i]);
    }
    
    if(not stationary) {
        deallocate_numa_vector<Integer_Type, Integer_Type>(&XI, XV_blks);
        deallocate_numa_vector<Integer_Type, Fractional_Type>(&XV, XV_blks);
        deallocate_numa_vector<Integer_Type, Integer_Type>(&YI, YI_blks);
        deallocate_numa_vector<Integer_Type, Fractional_Type>(&YV, YV_blks);
        for(int32_t i = 0; i < num_owned_segments; i++) {
            deallocate_numa_vector<Integer_Type, Integer_Type>(&YIt[i], YIt_blks[i]);
            deallocate_numa_vector<Integer_Type, Fractional_Type>(&YVt[i], YVt_blks[i]);
        }
        deallocate_numa_vector<Integer_Type, char>(&T, T_blks);
    }
}


#ifdef TIMING
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::times() {
    if(Env::is_master) {
        double sum = 0.0, mean = 0.0, std_dev = 0.0;
        std::cout << "INFO(rank=" << Env::rank << "): " << "Init     time: " << init_time[0] << " seconds" << std::endl;
        stats(bcast_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Bcast    time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        stats(combine_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Combine  time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        stats(apply_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Apply    time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        stats(convergence_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Converge time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        std::cout << "INFO(rank=" << Env::rank << "): " << "Execute time: " << execute_time[0] << " seconds" << std::endl;
        
        //std::cout << "DETAILED TIMING " << init_time[0] * 1e3;
        //stats(bcast_time, sum, mean, std_dev);
        //std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        //stats(combine_time, sum, mean, std_dev);
        //std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        //stats(apply_time, sum, mean, std_dev);
        //std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        //std::cout << " " << execute_time[0] * 1e3 << std::endl;
        
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev) {
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
}
#endif

#endif