/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include <type_traits>
#include <numeric>

#include "mpi/types.hpp" 

struct State { State() {}; };

enum Ordering_type
{
  _ROW_,
  _COL_
};   

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
class Vertex_Program
{
    public:
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph,
                        bool stationary_ = false, bool gather_depends_on_apply_ = false, 
                        bool apply_depends_on_iter_ = false, Ordering_type = _ROW_);
        ~Vertex_Program();
        Vertex_Methods_Impl Vertex_Methods;
        Vertex_Program(Vertex_Methods_Impl const &VMs) : Vertex_Methods(VMs) { };
        void set_root(Integer_Type root_) { 
            Vertex_Methods.set_root(root_);
        };
        
        void execute(Integer_Type num_iterations_ = 0);
        void initialize();
        template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_, typename Vertex_Methods_Impl_>
        void initialize(const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_, Vertex_Methods_Impl_> &VProgram);
        void free();
        void checksum();
        void display(Integer_Type count = 31);
        
        Integer_Type num_iterations = 0;
        bool stationary = false;
        bool gather_depends_on_apply = false;
        bool apply_depends_on_iter = false;
        Integer_Type iteration = 0;
        std::vector<Vertex_State> V;              // Values
        std::vector<std::vector<Vertex_State>> Vt;
        std::vector<std::vector<Integer_Type>> W; // Values (triangle counting)
    protected:
        bool already_initialized = false;
        bool check_for_convergence = false;
        bool converged = false;
        void init_stationary();
        void init_nonstationary();
        void init_stationary_postprocess();
        void init_nonstationary_postprocess();
        void bcast();
        void scatter_gather_stationary();
        void scatter_gather_nonstationary();
        void scatter_gather_nonstationary_activity_filtering();
        void scatter();
        void gather();
        void scatter_stationary();
        void gather_stationary();
        void scatter_nonstationary();
        void gather_nonstationary();
        void bcast_stationary();
        void bcast_stationary(int tid);
        void bcast_nonstationary();
        void bcast_nonstationary(int tid);
        void combine();
        void combine_2d_stationary();
        void combine_2d_stationary(int tid);
        void combine_2d_nonstationary();
        void combine_2d_nonstationary(int tid);
        void combine_postprocess();
        void combine_postprocess_stationary_for_all();
        void combine_postprocess_nonstationary_for_all();
        void apply();                        
        void apply_stationary();
        void apply_stationary(int tid);
        void apply_nonstationary();
        void apply_nonstationary(int tid);

        
        void spmv_stationary(struct Tile2D<Weight, Integer_Type, Fractional_Type>& tile,
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data); // Stationary spmv/spmspv
        
        
        void thread_function_stationary(int tid);
        void thread_function_nonstationary(int tid);
                
        void spmv_nonstationary(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile, // Stationary spmv/spmspv 
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data, 
                std::vector<Fractional_Type> &xv_data, 
                std::vector<Integer_Type> &xi_data,
                std::vector<char> &t_data);
        
        void wait_for_all();
        void wait_for_sends();
        void wait_for_recvs();
        bool has_converged();
        bool has_converged(int tid);
        Integer_Type get_vid(Integer_Type index, int32_t segment);
        Integer_Type get_vid(Integer_Type index);
        
        struct Triple<Weight, Integer_Type> tile_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                       struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> leader_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);  
        MPI_Comm communicator_info();  
        MPI_Comm communicator;               
        Ordering_type ordering_type;
        Tiling_type tiling_type;
        //Filtering_type filtering_type;
        Integer_Type nrows, ncols;
        uint32_t nrowgrps, ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nrowgrps, rank_ncolgrps;
        Integer_Type tile_height, tile_width;
        int32_t owned_segment, accu_segment_rg, accu_segment_cg, accu_segment_row, accu_segment_col;
        std::vector<int32_t> local_col_segments;
        std::vector<int32_t> accu_segment_col_vec;
        //std::vector<int32_t> accu_segment_row_vec;
        std::vector<int32_t> all_rowgrp_ranks_accu_seg;
        //std::vector<int32_t> accu_segment_rg_vec;
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
        MPI_Comm rowgrps_communicator;
        MPI_Comm colgrps_communicator;
        
        std::vector<MPI_Request> out_requests;
        std::vector<MPI_Request> in_requests;
        std::vector<MPI_Request> out_requests_;
        std::vector<MPI_Request> in_requests_;
        std::vector<MPI_Status> out_statuses;
        std::vector<MPI_Status> in_statuses;
        
        std::vector<std::vector<MPI_Request>> out_requests_t;
        std::vector<std::vector<MPI_Request>> in_requests_t;
        
    
        Matrix<Weight, Integer_Type, Fractional_Type>* A;          // Adjacency list
        

        
        
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        //std::vector<Integer_Type> nnz_cols_size_local;
        std::vector<int32_t> accu_segment_rows, accu_segment_cols;
        std::vector<int32_t> convergence_vec;
        
        
        std::vector<std::vector<Fractional_Type>> X;               // Messages 
        std::vector<std::vector<std::vector<Fractional_Type>>> Y;  // Accumulators
        //std::vector<std::vector<Fractional_Type>> Y1;  // Accumulators
        //std::vector<std::vector<Fractional_Type>> Yt;  // Accumulators
        std::vector<char> C;                                       // Convergence vector
        std::vector<std::vector<char>> Ct;
        /* Nonstationary */
        std::vector<std::vector<Integer_Type>> XI;                 // X Indices (Nonstationary)
        std::vector<std::vector<Fractional_Type>> XV;              // X Values  (Nonstationary)
        std::vector<std::vector<std::vector<Integer_Type>>> YI;    // Y Indices (Nonstationary)
        std::vector<std::vector<std::vector<Fractional_Type>>> YV; // Y Values (Nonstationary)
        std::vector<std::vector<char>> T;                          // Accumulators activity vectors
        std::vector<Integer_Type> msgs_activity_statuses;
        std::vector<std::vector<Integer_Type>> accus_activity_statuses;
        std::vector<Integer_Type> activity_statuses;
        std::vector<Integer_Type> active_vertices;
        /* Row/Col Filtering indices */
        std::vector<std::vector<char>>* I;
        std::vector<std::vector<Integer_Type>>* IV;
        std::vector<std::vector<char>>* J;
        std::vector<std::vector<Integer_Type>>* JV;
        
        
        std::vector<Integer_Type>* rowgrp_nnz_rows;
        std::vector<Integer_Type>* rowgrp_regular_rows;
        std::vector<Integer_Type>* rowgrp_source_rows;
        std::vector<Integer_Type>* colgrp_nnz_columns;
        std::vector<Integer_Type>* colgrp_sink_columns;
        
        std::vector<std::vector<Integer_Type>> rowgrp_nnz_rows_t;
        std::vector<std::vector<Integer_Type>> colgrp_nnz_cols_t;


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
        bool accu_activity_filtering = false;
        bool msgs_activity_filtering = false;
        
        bool broadcast_communication = true;
        bool incremental_accumulation = false;
        
        std::vector<MPI_Comm> rowgrps_communicators;
        std::vector<MPI_Comm> colgrps_communicators;
        
        pthread_barrier_t p_barrier;
        
        #ifdef TIMING
        void times();
        void stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev);
        std::vector<double> init_time;
        std::vector<double> bcast_time;
        std::vector<double> combine_time;
        std::vector<double> apply_time;
        std::vector<double> execute_time;
        #endif
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::Vertex_Program(
         Graph<Weight,Integer_Type, Fractional_Type> &Graph, bool stationary_, 
         bool gather_depends_on_apply_, bool apply_depends_on_iter_, Ordering_type ordering_type_)
{

    A = Graph.A;
    directed = A->directed;
    transpose = A->transpose;
    stationary = stationary_;
    gather_depends_on_apply = gather_depends_on_apply_;
    apply_depends_on_iter = apply_depends_on_iter_;
    ordering_type = ordering_type_;
    tiling_type = A->tiling->tiling_type;
    owned_segment = A->owned_segment;
    leader_ranks = A->leader_ranks;
    owned_segments = Graph.A->owned_segments;
    owned_segments_all = Graph.A->owned_segments_all;
    
    convergence_vec.resize(Env::nthreads);
    pthread_barrier_init(&p_barrier, NULL, Env::nthreads);

    if(ordering_type == _ROW_)
    {
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
        accu_segment_col = A->accu_segment_col;
        accu_segment_row = A->accu_segment_row;
        all_rowgrp_ranks_accu_seg = A->all_rowgrp_ranks_accu_seg;
        accu_segment_rg = A->accu_segment_rg;
        accu_segment_cg = A->accu_segment_cg;
        follower_rowgrp_ranks_rg = A->follower_rowgrp_ranks_rg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_rowgrp_ranks_accu_seg_rg;
        leader_ranks_cg = A->leader_ranks_cg;
        follower_colgrp_ranks_cg = A->follower_colgrp_ranks_cg;
        follower_colgrp_ranks = A->follower_colgrp_ranks;
        local_tiles_row_order = A->local_tiles_row_order;
        local_tiles_col_order = A->local_tiles_col_order;
        follower_rowgrp_ranks = A->follower_rowgrp_ranks;
        follower_rowgrp_ranks_accu_seg = A->follower_rowgrp_ranks_accu_seg;
        rowgrps_communicator = Env::rowgrps_comm;
        colgrps_communicator = Env::colgrps_comm;
        all_rowgrp_ranks = A->all_rowgrp_ranks;
        nnz_row_sizes_loc = A->nnz_row_sizes_loc;
        nnz_col_sizes_loc = A->nnz_col_sizes_loc;
        nnz_row_sizes_all = A->nnz_row_sizes_all;
        nnz_col_sizes_all = A->nnz_col_sizes_all;
        I = &(Graph.A->I);
        IV = &(Graph.A->IV);
        J = &(Graph.A->J);
        JV = &(Graph.A->JV);
        rowgrp_nnz_rows = &(Graph.A->rowgrp_nnz_rows);
        rowgrp_regular_rows = &(Graph.A->rowgrp_regular_rows);
        rowgrp_source_rows = &(Graph.A->rowgrp_source_rows);
        colgrp_nnz_columns = &(Graph.A->colgrp_nnz_columns);
        colgrp_sink_columns = &(Graph.A->colgrp_sink_columns);
        

        nnz_rows_sizes = A->nnz_rows_sizes;
        nnz_cols_sizes = A->nnz_cols_sizes;
        nnz_rows_size = A->nnz_rows_size;
        nnz_cols_size = A->nnz_cols_size;

        
        out_requests_t.resize(Env::nthreads);
        in_requests_t.resize(Env::nthreads);

        num_owned_segments = Graph.A->num_owned_segments;
        local_tiles_row_order_t = Graph.A->local_tiles_row_order_t;
        local_tiles_col_order_t = Graph.A->local_tiles_col_order_t;
        
        rowgrp_nnz_rows_t = Graph.A->rowgrp_nnz_rows_t;
        colgrp_nnz_cols_t = Graph.A->colgrp_nnz_cols_t;
        
        accu_segment_rows = Graph.A->accu_segment_rows;
        accu_segment_cols = Graph.A->accu_segment_cols;
        
        rowgrps_communicators = Env::rowgrps_comms;
        colgrps_communicators = Env::colgrps_comms;
        
    }
    else if (ordering_type == _COL_)
    {
        nrows = A->ncols;
        ncols = A->nrows;
        nrowgrps = A->ncolgrps;
        ncolgrps = A->nrowgrps;
        rowgrp_nranks = A->tiling->colgrp_nranks;
        colgrp_nranks = A->tiling->rowgrp_nranks;
        rank_nrowgrps = A->tiling->rank_ncolgrps;
        rank_ncolgrps = A->tiling->rank_nrowgrps;
        tile_height = A->tile_width;
        tile_width = A->tile_height;
        local_row_segments = A->local_col_segments;
        local_col_segments = A->local_row_segments;
        accu_segment_col = A->accu_segment_row;
        accu_segment_row = A->accu_segment_col;
        all_rowgrp_ranks_accu_seg = A->all_colgrp_ranks_accu_seg;
        accu_segment_rg = A->accu_segment_cg;
        accu_segment_cg = A->accu_segment_rg;
        follower_rowgrp_ranks_rg = A->follower_colgrp_ranks_cg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_colgrp_ranks_accu_seg_cg;
        leader_ranks_cg = A->leader_ranks_rg;
        follower_colgrp_ranks_cg = A->follower_rowgrp_ranks_rg;
        follower_colgrp_ranks = A->follower_rowgrp_ranks;
        local_tiles_row_order = A->local_tiles_col_order;
        local_tiles_col_order = A->local_tiles_row_order;
        follower_rowgrp_ranks = A->follower_colgrp_ranks;
        follower_rowgrp_ranks_accu_seg = A->follower_colgrp_ranks_accu_seg;
        rowgrps_communicator = Env::colgrps_comm;
        colgrps_communicator = Env::rowgrps_comm;
        all_rowgrp_ranks = A->all_colgrp_ranks;
        nnz_row_sizes_loc = A->nnz_col_sizes_loc;
        nnz_col_sizes_loc = A->nnz_row_sizes_loc;
        nnz_row_sizes_all = A->nnz_col_sizes_all;
        nnz_col_sizes_all = A->nnz_row_sizes_all;
        I = &(Graph.A->J);
        IV= &(Graph.A->JV);
        J = &(Graph.A->I);
        JV= &(Graph.A->IV);
        rowgrp_nnz_rows = &(Graph.A->colgrp_nnz_columns);
        rowgrp_regular_rows = &(Graph.A->rowgrp_regular_rows);
        rowgrp_source_rows = &(Graph.A->colgrp_sink_columns);
        colgrp_nnz_columns = &(Graph.A->rowgrp_nnz_rows);
        colgrp_sink_columns = &(Graph.A->rowgrp_source_rows);
        
        nnz_rows_sizes = Graph.A->nnz_cols_sizes;
        nnz_cols_sizes = Graph.A->nnz_rows_sizes;
        nnz_rows_size = Graph.A->nnz_cols_size;
        nnz_cols_size = Graph.A->nnz_rows_size;


        out_requests_t.resize(Env::nthreads);
        in_requests_t.resize(Env::nthreads);
 
        
        num_owned_segments = Graph.A->num_owned_segments;
        local_tiles_row_order_t = Graph.A->local_tiles_col_order_t;
        local_tiles_col_order_t = Graph.A->local_tiles_row_order_t;
        
        rowgrp_nnz_rows_t = Graph.A->colgrp_nnz_cols_t;
        colgrp_nnz_cols_t = Graph.A->rowgrp_nnz_rows_t;
        
        accu_segment_rows = Graph.A->accu_segment_cols;
        accu_segment_cols = Graph.A->accu_segment_rows;
        
        rowgrps_communicators = Env::colgrps_comms;
        colgrps_communicators = Env::rowgrps_comms;

    }   
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::free()
{
    V.clear();
    V.shrink_to_fit();

    C.clear();
    C.shrink_to_fit();
    
        
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            X[i].clear();
            X[i].shrink_to_fit();
        }
        

        for (uint32_t i = 0; i < rank_nrowgrps; i++)
        {
            if(local_row_segments[i] == owned_segment)
            {
                for(uint32_t j = 0; j < rowgrp_nranks; j++)
                {
                    Y[i][j].clear();
                    Y[i][j].shrink_to_fit();
                }
                    
            }
            else
            {
                Y[i][0].clear();
                Y[i][0].shrink_to_fit();
            }
            Y[i].clear();
            Y[i].shrink_to_fit();
        }   
        Y.clear();
        Y.shrink_to_fit();
    
    
    if(not stationary)
    {
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
        {
            XV[i].clear();
            XV[i].shrink_to_fit();
            XI[i].clear();
            XI[i].shrink_to_fit();
        }
        
        for(uint32_t i = 0; i < rank_nrowgrps; i++)
        {
          if(local_row_segments[i] == owned_segment)
            {
                for(uint32_t j = 0; j < rowgrp_nranks; j++)
                {
                    YV[i][j].clear();
                    YV[i][j].shrink_to_fit();
                    YI[i][j].clear();
                    YI[i][j].shrink_to_fit();
                }
            }
            else
            {
                YV[i][0].clear();
                YV[i][0].shrink_to_fit();
                YI[i][0].clear();
                YI[i][0].shrink_to_fit();
            }
            YV[i].clear();
            YV[i].shrink_to_fit();
            YI[i].clear();
            YI[i].shrink_to_fit();
        }
    }   
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::execute(Integer_Type num_iterations_) {
    
    num_iterations = num_iterations_;
    if(!num_iterations)
        check_for_convergence = true; 
    
    if(not already_initialized)
        initialize();
    

    
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    
    if(stationary) {
        //do{
            //bcast();
            std::vector<std::thread> threads;
            for(int i = 0; i < Env::nthreads; i++) {
                threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_stationary, this, i));
            }
            
            for(std::thread& th: threads) {
                th.join();
            }
            /*
            iteration++;
            Env::print_num("Iteration", iteration);
        } while(not has_converged());
        */
        
    }
    else {
        std::vector<std::thread> threads;
        for(int i = 0; i < Env::nthreads; i++) {
            threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_nonstationary, this, i));
        }
        
        for(std::thread& th: threads) {
            th.join();
        }
        /*
        while(true) {
            //bcast();
            combine();
            //apply();
            iteration++;
            Env::print_num("Iteration", iteration);
            if(check_for_convergence) {
                converged = has_converged();
                if(converged) {
                    break;
                }
            }
            else if(iteration >= num_iterations) {
                break;
            }
        }
        */
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::initialize() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    init_stationary();
    if(not stationary) {
        init_nonstationary();
    }
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_, typename Vertex_Methods_Impl_>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::initialize(
    const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_, Vertex_Methods_Impl_>& VProgram) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    init_stationary();
    if(stationary) {
        for(int32_t k = 0; k < num_owned_segments; k++) {
            uint32_t yi = accu_segment_rows[k];
            auto &i_data = (*I)[yi];         
            auto& v_data = Vt[k];
            auto& c_data = Ct[k];
            Integer_Type v_nitems = v_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = v_data[i]; 
                if(i_data[i])
                    c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[k]), state, (const State&) VProgram.Vt[k][i]);
            }
        }       
    }
    else {
        fprintf(stderr, "ERROR(rank=%d): Not implemented\n", Env::rank);
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_stationary() {
    // Initialize Values
    //if(stationary) {
        Vt.resize(num_owned_segments, std::vector<Vertex_State>(tile_width));
        Ct.resize(num_owned_segments, std::vector<char>(tile_width));
        for(int32_t k = 0; k < num_owned_segments; k++) {
            auto& v_data = Vt[k];
            auto& c_data = Ct[k];
            Integer_Type v_nitems = v_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = v_data[i]; 
                c_data[i] = Vertex_Methods.initializer(get_vid(i, owned_segments[k]), state);
                //c_data[i] = initializer(get_vid(i, owned_segments[k]), state);
            }
        }
        /*
    }
    else {
        V.resize(tile_width);
        Integer_Type v_nitems = V.size();
        C.resize(tile_width);
        //#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = V[i]; 
            C[i] = Vertex_Methods.initializer(get_vid(i), state);
        }        
    }
        */
    // Initialize messages
    std::vector<Integer_Type> x_sizes;   
    x_sizes = nnz_col_sizes_loc;
    X.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        X[i].resize(x_sizes[i]);

    // Initialiaze accumulators
    std::vector<Integer_Type> y_sizes;   
    y_sizes = nnz_row_sizes_loc;
    Y.resize(rank_nrowgrps);
    int32_t k = 0;
    for(uint32_t i = 0; i < rank_nrowgrps; i++) {
        if(leader_ranks[local_row_segments[i]] == Env::rank) {    
            Y[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                Y[i][j].resize(y_sizes[i]);
            k++;
        }
        else {
            Y[i].resize(1);
            Y[i][0].resize(y_sizes[i]);
        }
    }
    assert(k == num_owned_segments);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::init_nonstationary()
{
    //Integer_Type v_nitems = V.size();
    // Initialize activity statuses for all column groups
    // Assuming ncolgrps == nrowgrps
    activity_statuses.resize(ncolgrps);
    
    std::vector<Integer_Type> x_sizes;    
    //if(compression_type == _CSC_)
    //    x_sizes.resize(rank_ncolgrps, tile_height);
    //else if((compression_type == _DCSC_) or (compression_type == _TCSC_))
    //else    
        x_sizes = nnz_col_sizes_loc;
    
    // Initialize nonstationary messages values
    XV.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XV[i].resize(x_sizes[i]);
    // Initialize nonstationary messages indices
    XI.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XI[i].resize(x_sizes[i]);
    
    //msgs_activity_statuses.resize(colgrp_nranks);
    msgs_activity_statuses.resize(rank_ncolgrps);
    
    std::vector<Integer_Type> y_sizes;
    //if((compression_type == _CSC_) or (compression_type == _DCSC_))
    //    y_sizes.resize(rank_nrowgrps, tile_height);
    //else if(compression_type == _TCSC_)    
    //else    
        y_sizes = nnz_row_sizes_loc;
    
    T.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        T[i].resize(y_sizes[i]);
    
    accus_activity_statuses.resize(Env::nthreads, std::vector<Integer_Type>(rowgrp_nranks));
    
    // Initialiaze nonstationary accumulators values
    YV.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(leader_ranks[local_row_segments[i]] == Env::rank) {  
        //if(local_row_segments[i] == owned_segment)
        //{
            YV[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                YV[i][j].resize(y_sizes[i]);
        }
        else
        {
            YV[i].resize(1);
            YV[i][0].resize(y_sizes[i]);
        }
    }
    // Initialiaze nonstationary accumulators indices
    YI.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(leader_ranks[local_row_segments[i]] == Env::rank) {  
        //if(local_row_segments[i] == owned_segment)
        //{
            YI[i].resize(rowgrp_nranks);
            for(uint32_t j = 0; j < rowgrp_nranks; j++)
                YI[i][j].resize(y_sizes[i]);
        }
        else
        {
            YI[i].resize(1);
            YI[i][0].resize(y_sizes[i]);
        }
    }
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        uint32_t yi = i;
        uint32_t yo = 0;
        if(leader_ranks[local_row_segments[i]] == Env::rank)
        //if(local_row_segments[k] == owned_segment)
            yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        for(uint32_t j = 0; j < y_nitems; j++)
                y_data[j] = Vertex_Methods.infinity();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
        
    if(stationary) {
        scatter_gather_stationary();
        bcast_stationary();
    }
    else {
        scatter_gather_nonstationary();
        bcast_nonstationary();
    }
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Bcast", elapsed_time);
    bcast_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::scatter_gather_stationary() {
    for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t xo = accu_segment_cols[k];    
        std::vector<Fractional_Type>& x_data = X[xo];
        auto& JC = colgrp_nnz_cols_t[k];
        Integer_Type JC_nitems = JC.size();
        auto& v_data = Vt[k];
        //#pragma omp parallel for schedule(static)
        for(uint32_t j = 0; j < JC_nitems; j++) {
            Vertex_State& state = v_data[JC[j]];
            //x_data[j] = messenger(state);
            x_data[j] = Vertex_Methods.messenger(state);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::scatter_gather_nonstationary() {
    for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t xo = accu_segment_cols[k];
        std::vector<Fractional_Type> &x_data = X[xo];
        Integer_Type x_nitems = x_data.size();
        std::vector<Fractional_Type> &xv_data = XV[xo];
        std::vector<Integer_Type> &xi_data = XI[xo];
        auto& JC = colgrp_nnz_cols_t[k];
        Integer_Type JC_nitems = JC.size();
        auto& v_data = Vt[k];
        auto& c_data = Ct[k];
        Integer_Type i = 0;
        Integer_Type l = 0;
        ////#pragma omp parallel for schedule(static) Don't because of l
        for(Integer_Type j = 0; j < JC_nitems; j++) {
            i = JC[j];
            Vertex_State& state = v_data[i];
            if(c_data[i]) {
                //x_data[j] = messenger(state); 
                x_data[j] = Vertex_Methods.messenger(state); 
                xv_data[l] = x_data[j];
                xi_data[l] = j;
                l++;
            }
            else
                x_data[j] = Vertex_Methods.infinity();
        }
        
        
        //uint32_t xo = accu_segment_col;
        //std::vector<Fractional_Type>& x_data = X[xo];
        
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
            activity_statuses[owned_segments[k]] = msgs_activity_statuses[xo];
            
            //Env::barrier();
            for(int32_t r = 0; r < Env::nranks; r++) {
                //int32_t r = leader_ranks[i];
                
                if(r != Env::rank) {
                    int32_t j = k + (r * num_owned_segments);
                    int32_t m = owned_segments_all[j];
                    MPI_Sendrecv(&activity_statuses[owned_segments[k]], 1, TYPE_INT, r, k, 
                                 &activity_statuses[m], 1, TYPE_INT, r, k, Env::MPI_WORLD, MPI_STATUS_IGNORE);
                }
            }
            //Env::barrier();
            
        }

    }
    
    
    
    /*
    std::vector<Integer_Type> nnz_sizes_all_val_temp(Env::nsegments);
    std::vector<Integer_Type> nnz_sizes_all_pos_temp(Env::nsegments);
    for(int32_t k = 0; k < num_owned_segments; k++) {
        int32_t j = k + (Env::rank * num_owned_segments);
        nnz_sizes_all_val_temp[j] = activity_statuses[owned_segments[k]];
        nnz_sizes_all_pos_temp[j] = owned_segments[k];
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
    */
    
    
    
    
    /*
    Integer_Type k = 0;
    auto& JC = colgrp_nnz_cols_t[0];
    Integer_Type JC_nitems = JC.size();
    Integer_Type i = 0;
    //#pragma omp parallel for schedule(static)
    for(Integer_Type j = 0; j < JC_nitems; j++) {
        i = JC[j];
        Vertex_State &state = V[i];
        if(C[i]) {
            //x_data[j] = messenger(state); 
            x_data[j] = Vertex_Methods.messenger(state); 
            xv_data[k] = x_data[j];
            xi_data[k] = j;
            k++;
        }
        else
            x_data[j] = Vertex_Methods.infinity();
    }
    */

    /*
    if(activity_filtering) {
        msgs_activity_statuses[xo] = k;
        scatter_gather_nonstationary_activity_filtering();
    }
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::scatter_gather_nonstationary_activity_filtering() {
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type>& x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    if(activity_filtering) {
        int nitems = msgs_activity_statuses[xo];
        // 0 all, 1 nothing, else nitems
        double ratio = (double) nitems/x_nitems;
        if(ratio <= activity_filtering_ratio)
            nitems++;
        else
            nitems = 0;
        msgs_activity_statuses[xo] = nitems;
        activity_statuses[owned_segment] = msgs_activity_statuses[xo];
        Env::barrier();
        for(int32_t i = 0; i < Env::nranks; i++) {
            int32_t r = leader_ranks[i];
            if(r != Env::rank)
                MPI_Sendrecv(&activity_statuses[owned_segment], 1, TYPE_INT, r, Env::rank, 
                             &activity_statuses[i], 1, TYPE_INT, r, r, Env::MPI_WORLD, MPI_STATUS_IGNORE);
        }
        Env::barrier();
    }
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
    std::vector<Fractional_Type>& x_data = X[xo];
    auto& JC = colgrp_nnz_cols_t[tid];
    Integer_Type JC_nitems = JC.size();
    auto& v_data = Vt[tid];
    for(uint32_t j = 0; j < JC_nitems; j++) {
        Vertex_State& state = v_data[JC[j]];
        x_data[j] = Vertex_Methods.messenger(state);
    }
    
    const int32_t col_chunk_size = rank_ncolgrps / Env::nthreads;
    const int32_t col_start = tid * col_chunk_size;
    const int32_t col_end = (tid != Env::nthreads - 1) ? col_start + col_chunk_size : rank_ncolgrps;
    
    MPI_Request request;
    int32_t leader, col_group;
    for(int32_t i = col_start; i < col_end; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];
        std::vector<Fractional_Type>& xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicators[tid], &request);
        out_requests_t[tid].push_back(request);
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
     
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_stationary() {    
    MPI_Request request;
    int32_t leader;
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        int32_t col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];
        std::vector<Fractional_Type> &xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicator, &request);
        out_requests.push_back(request);
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();   
} 

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_nonstationary() {
    
    MPI_Request request;
    int32_t leader_cg;
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        leader_cg = leader_ranks_cg[local_col_segments[i]]; 
        std::vector<Fractional_Type>& xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        std::vector<Integer_Type>& xij_data = XI[i];
        std::vector<Fractional_Type>& xvj_data = XV[i];
        int nitems = 0;
        if(Env::rank_cg == leader_cg)
            nitems = msgs_activity_statuses[i];
        MPI_Ibcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicator, &request);
        MPI_Wait(&request, MPI_STATUSES_IGNORE);
        if(Env::rank_cg != leader_cg)
            msgs_activity_statuses[i] = nitems;
        if(activity_filtering and nitems) {
            if(nitems > 1) {
                MPI_Ibcast(xij_data.data(), nitems - 1, TYPE_INT, leader_cg, colgrps_communicator, &request);
                out_requests.push_back(request);
                MPI_Ibcast(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
                out_requests.push_back(request);
            }
        }
        else {
            MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
            out_requests.push_back(request);
        }
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();     
}   


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine() {        
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        combine_2d_stationary();
        //combine_postprocess();
    }
    else {
        combine_2d_nonstationary();
        //combine_postprocess();
    }
    #ifdef TIMING    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Combine", elapsed_time);
    combine_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_stationary(int tid) {
    int ret = Env::set_thread_affinity(tid);
    //bool converged = false;
    
    do {
        
        bcast_stationary(tid);
        combine_2d_stationary(tid);
        apply_stationary(tid);
    /* 
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    
    
    
    //for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t xo = accu_segment_cols[tid];    
        std::vector<Fractional_Type>& x_data = X[xo];
        auto& JC = colgrp_nnz_cols_t[tid];
        Integer_Type JC_nitems = JC.size();
        auto& v_data = Vt[tid];
        //#pragma omp parallel for schedule(static)
        for(uint32_t j = 0; j < JC_nitems; j++) {
            Vertex_State& state = v_data[JC[j]];
            //x_data[j] = messenger(state);
            x_data[j] = Vertex_Methods.messenger(state);
        }
    //}
    
    
    const int32_t col_chunk_size = rank_ncolgrps / Env::nthreads;
    const int32_t col_start = tid * col_chunk_size;
    const int32_t col_end = (tid != Env::nthreads - 1) ? col_start + col_chunk_size : rank_ncolgrps;
    
    MPI_Request request;
    int32_t leader, col_group;
    for(int32_t i = col_start; i < col_end; i++) {
        col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];
        std::vector<Fractional_Type>& xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicators[tid], &request);
        out_requests_t[tid].push_back(request);
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    */
    
    
    /*
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();

    #endif
    
    
    
    
    
    
    //if(!Env::rank and tid == 10)
      //  printf("%d/%d\n", tid, sched_getcpu());
    MPI_Request request;
    //uint32_t xi= 0, yi = 0, yo = 0, follower = 0, accu = 0, tile_th = 0, pair_idx = 0;
    //bool vec_owner = false, communication = false;
    uint32_t tile_th, pair_idx;
    int32_t leader;
    int32_t follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
    //for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        if(ordering_type == _ROW_)
            yi = tile.ith;
        else 
            yi = tile.jth;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        
        std::vector<Fractional_Type> &x_data = X[xi];
        
        const uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
        if(nnz) {
            #ifdef HAS_WEIGHT
            const Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
            #endif

            const Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
            const Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
            const Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;  
            
            if(ordering_type == _ROW_) {
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
            else {
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        
                        #ifdef HAS_WEIGHT
                        Vertex_Methods.combiner(y_data[j], x_data[IA[i]], A[i]);   
                        #else
                        Vertex_Methods.combiner(y_data[j], x_data[IA[i]]);
                        #endif
                    }
                } 
            }
        }
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            //MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                    follower = follower_rowgrp_ranks_rg[j];
                    accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                    Integer_Type yj_nitems = yj_data.size();
                    MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[tid], &request);
                    //in_requests.push_back(request);
                    in_requests_t[tid].push_back(request);
                }
            }
            else {
                MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[tid], &request);
                //out_requests.push_back(request);
                out_requests_t[tid].push_back(request);
            }
            xi = 0;
        }
    }
    
    
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();
        
    yi  = accu_segment_rows[tid];
    yo = accu_segment_rg;
    std::vector<Fractional_Type>& y_data = Y[yi][yo];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        accu = follower_rowgrp_ranks_accu_seg_rg[j];
        std::vector<Fractional_Type>& yj_data = Y[yi][accu];
        Integer_Type yj_nitems = yj_data.size();
        for(uint32_t i = 0; i < yj_nitems; i++)
            Vertex_Methods.combiner(y_data[i], yj_data[i]);
    }
   
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    
    #ifdef TIMING    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    if(Env::is_master and tid == 0) {
        Env::print_time("Combine", elapsed_time);
        combine_time.push_back(elapsed_time);
    }

    t1 = Env::clock();
    #endif
    */
    
    //for(int32_t k = 0; k < num_owned_segments; k++) {
        //uint32_t accu = 0;
        /*
        uint32_t yi = 0, yo = 0;
        yi  = accu_segment_rows[tid];
        yo = accu_segment_rg;
        //std::vector<Fractional_Type>& 
        auto& y_data = Y[yi][yo];
        auto& i_data = (*I)[yi];
        auto& iv_data = (*IV)[yi];
        auto& v_data = Vt[tid];
        auto& c_data = Ct[tid];
        Integer_Type v_nitems = v_data.size();
        //#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = v_data[i];
            if(i_data[i]) {
                c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
            }
            else
                c_data[i] = Vertex_Methods.applicator(state);
            
        }   
    //}
    
    
    int32_t start_row = tid * (rank_nrowgrps / Env::nthreads);
    int32_t end_row = (tid != Env::nthreads - 1) start + (rank_nrowgrps / Env::nthreads) ? rank_nrowgrps;
    //end = (tid != Env::nthreads - 1) ? end : rank_nrowgrps;
    for(int32_t i = start_row; i < end_row; i++) {
        for(uint32_t j = 0; j < Y[i].size(); j++) {
            std::vector<Fractional_Type> &y_data = Y[i][j];
            Integer_Type y_nitems = y_data.size();
            std::fill(y_data.begin(), y_data.end(), 0);
        }
    }
    */
    /*
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    if(Env::is_master and tid == 0) {
        Env::print_time("Apply", elapsed_time);
        apply_time.push_back(elapsed_time);
    }
    #endif
    
    */
    
        
    /*
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;    
    if(check_for_convergence) {
        convergence_vec[tid] = 0;     
        c_data = Ct[tid];
        Integer_Type c_nitems = c_data.size();   
        for(uint32_t i = 0; i < c_nitems; i++) {
            if(not c_data[i]) 
                c_sum_local++;
        }
        if(c_sum_local == c_nitems)
            convergence_vec[tid] = 1;
    }
    
    pthread_barrier_wait(&p_barrier);
    if(tid == 0) {
        iteration++;
        Env::print_num("Iteration", iteration);
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
    */

    }while(not has_converged(tid));   
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
    //uint32_t xi= 0, yi = 0, yo = 0, follower = 0, accu = 0, tile_th = 0, pair_idx = 0;
    //bool vec_owner = false, communication = false;
    uint32_t tile_th, pair_idx;
    int32_t leader;
    int32_t follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
    //for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        if(ordering_type == _ROW_)
            yi = tile.ith;
        else 
            yi = tile.jth;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        
        std::vector<Fractional_Type> &x_data = X[xi];
        
        const uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnz;
        if(nnz) {
            #ifdef HAS_WEIGHT
            const Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
            #endif

            const Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
            const Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
            const Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;  
            
            if(ordering_type == _ROW_) {
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
            else {
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        
                        #ifdef HAS_WEIGHT
                        Vertex_Methods.combiner(y_data[j], x_data[IA[i]], A[i]);   
                        #else
                        Vertex_Methods.combiner(y_data[j], x_data[IA[i]]);
                        #endif
                    }
                } 
            }
        }
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            //MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                    follower = follower_rowgrp_ranks_rg[j];
                    accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                    Integer_Type yj_nitems = yj_data.size();
                    MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[tid], &request);
                    //in_requests.push_back(request);
                    in_requests_t[tid].push_back(request);
                }
            }
            else {
                MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[tid], &request);
                //out_requests.push_back(request);
                out_requests_t[tid].push_back(request);
            }
            xi = 0;
        }
    }
    
    
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();
        
    yi  = accu_segment_rows[tid];
    yo = accu_segment_rg;
    std::vector<Fractional_Type>& y_data = Y[yi][yo];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        accu = follower_rowgrp_ranks_accu_seg_rg[j];
        std::vector<Fractional_Type>& yj_data = Y[yi][accu];
        Integer_Type yj_nitems = yj_data.size();
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
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_stationary() {

    int nthreads = Env::nthreads;
    std::vector<std::thread> threads;
    for(int i = 0; i < nthreads; i++) {
        threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_stationary, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }

}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::bcast_nonstationary(int tid) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    if(tid == 0) {
        t1 = Env::clock();
    }
    #endif
    
    //for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t xo = accu_segment_cols[tid];
        std::vector<Fractional_Type> &x_data = X[xo];
        Integer_Type x_nitems = x_data.size();
        std::vector<Fractional_Type> &xv_data = XV[xo];
        std::vector<Integer_Type> &xi_data = XI[xo];
        auto& JC = colgrp_nnz_cols_t[tid];
        Integer_Type JC_nitems = JC.size();
        auto& v_data = Vt[tid];
        auto& c_data = Ct[tid];
        Integer_Type i = 0;
        Integer_Type l = 0;
        ////#pragma omp parallel for schedule(static) Don't because of l
        for(Integer_Type j = 0; j < JC_nitems; j++) {
            i = JC[j];
            Vertex_State& state = v_data[i];
            if(c_data[i]) {
                //x_data[j] = messenger(state); 
                x_data[j] = Vertex_Methods.messenger(state); 
                xv_data[l] = x_data[j];
                xi_data[l] = j;
                l++;
            }
            else
                x_data[j] = Vertex_Methods.infinity();
        }
        
        
        //uint32_t xo = accu_segment_col;
        //std::vector<Fractional_Type>& x_data = X[xo];
        
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
            
            //Env::barrier();
            for(int32_t r = 0; r < Env::nranks; r++) {
                //int32_t r = leader_ranks[i];
                
                if(r != Env::rank) {
                    int32_t j = tid + (r * num_owned_segments);
                    int32_t m = owned_segments_all[j];
                    MPI_Sendrecv(&activity_statuses[owned_segments[tid]], 1, TYPE_INT, r, tid, 
                                 &activity_statuses[m], 1, TYPE_INT, r, tid, Env::MPI_WORLD, MPI_STATUS_IGNORE);
                }
            }
            //Env::barrier();
            
        }

        
    //}
    
    
    const int32_t col_chunk_size = rank_ncolgrps / Env::nthreads;
    const int32_t col_start = tid * col_chunk_size;
    const int32_t col_end = (tid != Env::nthreads - 1) ? col_start + col_chunk_size : rank_ncolgrps;
        
    MPI_Request request;
    int32_t leader_cg;
    for(int32_t i = col_start; i < col_end; i++) {
        leader_cg = leader_ranks_cg[local_col_segments[i]]; 
        std::vector<Fractional_Type>& xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        std::vector<Integer_Type>& xij_data = XI[i];
        std::vector<Fractional_Type>& xvj_data = XV[i];
        int nitems = 0;
        if(Env::rank_cg == leader_cg)
            nitems = msgs_activity_statuses[i];
        MPI_Ibcast(&nitems, 1, TYPE_INT, leader_cg, colgrps_communicators[tid], &request);
        MPI_Wait(&request, MPI_STATUSES_IGNORE);
        if(Env::rank_cg != leader_cg)
            msgs_activity_statuses[i] = nitems;
        if(activity_filtering and nitems) {
            if(nitems > 1) {
                MPI_Ibcast(xij_data.data(), nitems - 1, TYPE_INT, leader_cg, colgrps_communicators[tid], &request);
                out_requests_t[tid].push_back(request);
                MPI_Ibcast(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicators[tid], &request);
                out_requests_t[tid].push_back(request);
            }
        }
        else {
            MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicators[tid], &request);
            out_requests_t[tid].push_back(request);
        }
    }
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();     
    pthread_barrier_wait(&p_barrier);
    

    /*    
    std::fill(accus_activity_statuses[tid].begin(), accus_activity_statuses[tid].end(), 0);
    
   std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
    int32_t chunk_size = rank_ncolgrps / Env::nthreads;
    int32_t start = tid * chunk_size;
    int32_t end = (tid != Env::nthreads - 1) ? start + chunk_size : rank_ncolgrps;
    for(int32_t i = start; i < end; i++) 
        msgs_activity_statuses[i] = 0;   
    */
    
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
    //for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        
        if(ordering_type == _ROW_)
            yi = tile.ith;
        else 
            yi = tile.jth;
        
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        std::vector<Fractional_Type>& y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        std::vector<Fractional_Type>& x_data = X[xi];
        std::vector<Fractional_Type>& xv_data = XV[xi];
        std::vector<Integer_Type>& xi_data = XI[xi];
        std::vector<char>& t_data = T[yi];
        
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
        

        //if(tile.nedges)
        //    spmv_nonstationary(tile, y_data, x_data, xv_data, xi_data, t_data);
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            //MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
                    //if(Env::comm_split) {   
                        follower = follower_rowgrp_ranks_rg[j];
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    //}
                    //else {
                      //  follower = follower_rowgrp_ranks[j];
                        //accu = follower_rowgrp_ranks_accu_seg[j];
                    //}
                    if(activity_filtering and activity_statuses[tile.rg]) {
                        // 0 all / 1 nothing / else nitems 
                        int nitems = 0;
                        MPI_Status status;
                        MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, rowgrps_communicators[tid], &status);
                        accus_activity_statuses[tid][accu] = nitems;
                        if(accus_activity_statuses[tid][accu] > 1) {
                            std::vector<Integer_Type> &yij_data = YI[yi][accu];
                            std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                            MPI_Irecv(yij_data.data(), accus_activity_statuses[tid][accu] - 1, TYPE_INT, follower, pair_idx, rowgrps_communicators[tid], &request);
                            //in_requests.push_back(request);
                            in_requests_t[tid].push_back(request);
                            MPI_Irecv(yvj_data.data(), accus_activity_statuses[tid][accu] - 1, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[tid], &request);
                            //in_requests_.push_back(request);
                            in_requests_t[tid].push_back(request);
                        }
                    }
                    else {                                
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = yj_data.size();
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicators[tid], &request);
                        //in_requests.push_back(request);
                        in_requests_t[tid].push_back(request);
                    }       
                }   
            }
            else
            {
                std::vector<Integer_Type> &yi_data = YI[yi][yo];
                std::vector<Fractional_Type> &yv_data = YV[yi][yo];
                int nitems = 0;
                if(activity_filtering and activity_statuses[tile.rg]) {
                    std::vector<char> &t_data = T[yi];
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
                    MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, rowgrps_communicators[tid]);
                    if(nitems > 1) {
                        MPI_Isend(yi_data.data(), nitems - 1, TYPE_INT, leader, pair_idx, rowgrps_communicators[tid], &request);
                        //out_requests.push_back(request);
                        out_requests_t[tid].push_back(request);
                        MPI_Isend(yv_data.data(), nitems - 1, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[tid], &request);
                        //out_requests_.push_back(request);
                        out_requests_t[tid].push_back(request);
                    }
                }
                else {
                    MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicators[tid], &request);
                    //out_requests.push_back(request);
                    out_requests_t[tid].push_back(request);
                }
            }
            xi = 0;
            //yi++;
        }
    }
    
    
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();
    

    
    
    //uint32_t accu = 0;
    yi = accu_segment_rows[tid];
    yo = accu_segment_rg;
    std::vector<Fractional_Type>& y_data = Y[yi][yo];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        //if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        //else
          //  accu = follower_rowgrp_ranks_accu_seg[j];
        if(activity_filtering and accus_activity_statuses[tid][accu]) {
            if(accus_activity_statuses[tid][accu] > 1) {
                std::vector<Integer_Type>& yij_data = YI[yi][accu];
                std::vector<Fractional_Type>& yvj_data = YV[yi][accu];
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < accus_activity_statuses[tid][accu] - 1; i++) {
                    Integer_Type k = yij_data[i];
                    Vertex_Methods.combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else {
            std::vector<Fractional_Type>& yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            //#pragma omp parallel for schedule(static)
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
    
    uint32_t yi = 0, yo = 0;
     if(apply_depends_on_iter)
    {
        if(iteration == 0)
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                Integer_Type v_nitems = v_data.size();
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]], iteration);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            //}
        }
        else
        {
          //  for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                //Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[tid];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                //#pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State& state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l], iteration);
                }
         //   }
        }
    }
    else {
        if(iteration == 0)
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                Integer_Type v_nitems = v_data.size();
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State& state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            //}
        }
        else
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[tid];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                //#pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State &state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l]);
                }
            //}
        }
    }

            
    int32_t start = tid * (rank_nrowgrps / Env::nthreads);
    int32_t end = start + (rank_nrowgrps / Env::nthreads);
    end = (tid != Env::nthreads - 1) ? end : rank_nrowgrps;
    if(not gather_depends_on_apply and not apply_depends_on_iter) {
        for(int32_t i = start; i < end; i++) {
            for(uint32_t j = 0; j < Y[i].size(); j++) {
                std::vector<Fractional_Type> &y_data = Y[i][j];
                Integer_Type y_nitems = y_data.size();
                std::fill(y_data.begin(), y_data.end(), 0);
            }
        }
    }

    if(activity_filtering) {
        for(int32_t i = start; i < end; i++) {
            std::vector<char> &t_data = T[i];
            Integer_Type t_nitems = t_data.size();
            std::fill(t_data.begin(), t_data.end(), 0);
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
    do {
        bcast_nonstationary(tid);
        combine_2d_nonstationary(tid);
        apply_nonstationary(tid);

        
        
    }while(not has_converged(tid));
    
    /*
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    int ret = Env::set_thread_affinity(tid);
    
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order_t[tid]) {
    //for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        
        if(ordering_type == _ROW_)
            yi = tile.ith;
        else 
            yi = tile.jth;
        
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        std::vector<Fractional_Type>& y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        std::vector<Fractional_Type>& x_data = X[xi];
        std::vector<Fractional_Type>& xv_data = XV[xi];
        std::vector<Integer_Type>& xi_data = XI[xi];
        std::vector<char>& t_data = T[yi];
        
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
        

        //if(tile.nedges)
        //    spmv_nonstationary(tile, y_data, x_data, xv_data, xi_data, t_data);
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            //MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
                    //if(Env::comm_split) {   
                        follower = follower_rowgrp_ranks_rg[j];
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    //}
                    //else {
                      //  follower = follower_rowgrp_ranks[j];
                        //accu = follower_rowgrp_ranks_accu_seg[j];
                    //}
                    if(activity_filtering and activity_statuses[tile.rg]) {
                        // 0 all / 1 nothing / else nitems 
                        int nitems = 0;
                        MPI_Status status;
                        MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, rowgrps_communicator, &status);
                        accus_activity_statuses[tid][accu] = nitems;
                        if(accus_activity_statuses[tid][accu] > 1) {
                            std::vector<Integer_Type> &yij_data = YI[yi][accu];
                            std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                            MPI_Irecv(yij_data.data(), accus_activity_statuses[tid][accu] - 1, TYPE_INT, follower, pair_idx, rowgrps_communicator, &request);
                            //in_requests.push_back(request);
                            in_requests_t[tid].push_back(request);
                            MPI_Irecv(yvj_data.data(), accus_activity_statuses[tid][accu] - 1, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicator, &request);
                            //in_requests_.push_back(request);
                            in_requests_t[tid].push_back(request);
                        }
                    }
                    else {                                
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = yj_data.size();
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, rowgrps_communicator, &request);
                        //in_requests.push_back(request);
                        in_requests_t[tid].push_back(request);
                    }       
                }   
            }
            else
            {
                std::vector<Integer_Type> &yi_data = YI[yi][yo];
                std::vector<Fractional_Type> &yv_data = YV[yi][yo];
                int nitems = 0;
                if(activity_filtering and activity_statuses[tile.rg]) {
                    std::vector<char> &t_data = T[yi];
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
                    MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, rowgrps_communicator);
                    if(nitems > 1) {
                        MPI_Isend(yi_data.data(), nitems - 1, TYPE_INT, leader, pair_idx, rowgrps_communicator, &request);
                        //out_requests.push_back(request);
                        out_requests_t[tid].push_back(request);
                        MPI_Isend(yv_data.data(), nitems - 1, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicator, &request);
                        //out_requests_.push_back(request);
                        out_requests_t[tid].push_back(request);
                    }
                }
                else {
                    MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, rowgrps_communicator, &request);
                    //out_requests.push_back(request);
                    out_requests_t[tid].push_back(request);
                }
            }
            xi = 0;
            //yi++;
        }
    }
    
    
    MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    in_requests_t[tid].clear();
    

    
    
    //uint32_t accu = 0;
    yi = accu_segment_rows[tid];
    yo = accu_segment_rg;
    std::vector<Fractional_Type>& y_data = Y[yi][yo];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        //if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        //else
          //  accu = follower_rowgrp_ranks_accu_seg[j];
        if(activity_filtering and accus_activity_statuses[tid][accu]) {
            if(accus_activity_statuses[tid][accu] > 1) {
                std::vector<Integer_Type>& yij_data = YI[yi][accu];
                std::vector<Fractional_Type>& yvj_data = YV[yi][accu];
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < accus_activity_statuses[tid][accu] - 1; i++) {
                    Integer_Type k = yij_data[i];
                    Vertex_Methods.combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else {
            std::vector<Fractional_Type>& yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                Vertex_Methods.combiner(y_data[i], yj_data[i]);
        }
    }
    
    MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
    out_requests_t[tid].clear();
    

    std::fill(accus_activity_statuses[tid].begin(), accus_activity_statuses[tid].end(), 0);
    */
    
    /*
    int32_t chunk_size = rank_ncolgrps / Env::nthreads;
    int32_t start = tid * chunk_size;
    int32_t end = (tid != Env::nthreads - 1) ? start + chunk_size : rank_ncolgrps;
    for(int32_t i = start; i < end; i++) 
        msgs_activity_statuses[i] = 0;
    */

    /*
    #ifdef TIMING    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    if(Env::is_master and tid == 0) {
        Env::print_time("Combine", elapsed_time);
        combine_time.push_back(elapsed_time);
    }

    t1 = Env::clock();
    #endif    
    
    if(apply_depends_on_iter)
    {
        if(iteration == 0)
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                Integer_Type v_nitems = v_data.size();
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]], iteration);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            //}
        }
        else
        {
          //  for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                //Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[tid];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                //#pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State& state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l], iteration);
                }
         //   }
        }
    }
    else {
        if(iteration == 0)
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                Integer_Type v_nitems = v_data.size();
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State& state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            //}
        }
        else
        {
            //for(int32_t k = 0; k < num_owned_segments; k++) {
                yi  = accu_segment_rows[tid];
                yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[tid];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[tid];
                auto& c_data = Ct[tid];
                //#pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State &state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l]);
                }
            //}
        }
    }

            
    int32_t start = tid * (rank_nrowgrps / Env::nthreads);
    int32_t end = start + (rank_nrowgrps / Env::nthreads);
    end = (tid != Env::nthreads - 1) ? end : rank_nrowgrps;
    if(not gather_depends_on_apply and not apply_depends_on_iter) {
        for(int32_t i = start; i < end; i++) {
            for(uint32_t j = 0; j < Y[i].size(); j++) {
                std::vector<Fractional_Type> &y_data = Y[i][j];
                Integer_Type y_nitems = y_data.size();
                std::fill(y_data.begin(), y_data.end(), 0);
            }
        }
    }

    if(activity_filtering) {
        for(int32_t i = start; i < end; i++) {
            std::vector<char> &t_data = T[i];
            Integer_Type t_nitems = t_data.size();
            std::fill(t_data.begin(), t_data.end(), 0);
        }
    }
    
    
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    if(Env::is_master and tid == 0) {
        Env::print_time("Apply", elapsed_time);
        apply_time.push_back(elapsed_time);
    }
    #endif
    */
    //has_converged()
    /*
    auto chunk_size = msgs_activity_statuses.size() / Env::nthreads;
    auto start = msgs_activity_statuses.begin() + (tid * chunk_size);
    auto end = (tid != Env::nthreads - 1) ? start + chunk_size : msgs_activity_statuses.end();
    std::fill(start, end, 0);
    */
    
    
    
    /*
    convergence_vec[tid] = 0; 
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;
    auto& c_data = Ct[tid];
    Integer_Type c_nitems = c_data.size();   
    for(uint32_t i = 0; i < c_nitems; i++) {
        if(not c_data[i]) 
            c_sum_local++;
    }
    if(c_sum_local == c_nitems)
        convergence_vec[tid] = 1;
    */
    /*
    if(tid == 0) {
        if(std::sum(convergence_vec.begin(), convergence_vec.end(), 0) == Env::nthreads)
            c_sum_local = 1;
        else 
            c_sum_local = 0;
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(c_sum_gloabl == (uint64_t) Env::nranks)
            converged = true;
    }
    convergence_vec[tid] = 0;     
    */    
    //return(converged); 
    
    
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_2d_nonstationary() {
    
    int nthreads = Env::nthreads;
    std::vector<std::thread> threads;
    for(int i = 0; i < nthreads; i++) {
        threads.push_back(std::thread(&Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::thread_function_nonstationary, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
    
    std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
    
}



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_postprocess() {
    if(stationary)
        combine_postprocess_stationary_for_all();
    else {
        combine_postprocess_nonstationary_for_all();
        
        std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
        std::fill(accus_activity_statuses.begin(), accus_activity_statuses.end(), 0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_postprocess_stationary_for_all()
{
    for(int32_t k = 0; k < num_owned_segments; k++) {    
        MPI_Waitall(in_requests_t[k].size(), in_requests_t[k].data(), MPI_STATUSES_IGNORE);
        in_requests_t[k].clear();
    }
    
    for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t accu = 0;
        uint32_t yi  = accu_segment_rows[k];
        uint32_t yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                Vertex_Methods.combiner(y_data[i], yj_data[i]);
        }
    }

    for(int32_t k = 0; k < num_owned_segments; k++) {    
        MPI_Waitall(out_requests_t[k].size(), out_requests_t[k].data(), MPI_STATUSES_IGNORE);
        out_requests_t[k].clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::combine_postprocess_nonstationary_for_all() {
    wait_for_recvs();
    uint32_t accu = 0;
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type>& y_data = Y[yi][yo];
    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
        if(Env::comm_split)
            accu = follower_rowgrp_ranks_accu_seg_rg[j];
        else
            accu = follower_rowgrp_ranks_accu_seg[j];
        if(activity_filtering and accus_activity_statuses[accu]) {
            if(accus_activity_statuses[accu] > 1) {
                std::vector<Integer_Type> &yij_data = YI[yi][accu];
                std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < accus_activity_statuses[accu] - 1; i++) {
                    Integer_Type k = yij_data[i];
                    Vertex_Methods.combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else {
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                Vertex_Methods.combiner(y_data[i], yj_data[i]);
        }
    }
    wait_for_sends();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::wait_for_all() {
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    if(not stationary and activity_filtering) {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::wait_for_sends() {
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    if(not stationary and activity_filtering) {
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::wait_for_recvs() {
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    if(not stationary and activity_filtering) {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        apply_stationary();
    }
    else {
        apply_nonstationary();
    }

    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Apply", elapsed_time);
    apply_time.push_back(elapsed_time);
    #endif
    //MPI_Barrier(MPI_COMM_WORLD);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_stationary(int tid) {
    uint32_t yi  = accu_segment_rows[tid];
    uint32_t yo = accu_segment_rg;
    auto& y_data = Y[yi][yo];
    auto& i_data = (*I)[yi];
    auto& iv_data = (*IV)[yi];
    auto& v_data = Vt[tid];
    auto& c_data = Ct[tid];
    Integer_Type v_nitems = v_data.size();
    for(uint32_t i = 0; i < v_nitems; i++) {
        Vertex_State &state = v_data[i];
        if(i_data[i]) {
            c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
        }
        else
            c_data[i] = Vertex_Methods.applicator(state);
    }   

    int32_t start_row = tid * (rank_nrowgrps / Env::nthreads);
    int32_t end_row = (tid != Env::nthreads - 1) ? start_row + (rank_nrowgrps / Env::nthreads) : rank_nrowgrps;
    for(int32_t i = start_row; i < end_row; i++) {
        for(uint32_t j = 0; j < Y[i].size(); j++) {
            std::vector<Fractional_Type> &y_data = Y[i][j];
            Integer_Type y_nitems = y_data.size();
            std::fill(y_data.begin(), y_data.end(), 0);
        }
    }
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_stationary() {
    for(int32_t k = 0; k < num_owned_segments; k++) {
        uint32_t accu = 0;
        uint32_t yi  = accu_segment_rows[k];
        uint32_t yo = accu_segment_rg;
        std::vector<Fractional_Type>& y_data = Y[yi][yo];
        auto& i_data = (*I)[yi];
        auto& iv_data = (*IV)[yi];
        auto& v_data = Vt[k];
        auto& c_data = Ct[k];
        Integer_Type v_nitems = v_data.size();
        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State &state = v_data[i];
            if(i_data[i]) {
                c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
            }
            else
                c_data[i] = Vertex_Methods.applicator(state);
            
        }   
    }
    
    for(uint32_t i = 0; i < rank_nrowgrps; i++) {
        for(uint32_t j = 0; j < Y[i].size(); j++) {
            std::vector<Fractional_Type> &y_data = Y[i][j];
            Integer_Type y_nitems = y_data.size();
            std::fill(y_data.begin(), y_data.end(), 0);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::apply_nonstationary()
{

    
    

    
    //uint32_t yi = accu_segment_row;
    //uint32_t yo = accu_segment_rg;
    //std::vector<Fractional_Type> &y_data = Y[yi][yo];

    //Integer_Type v_nitems = V.size();
    
    if(apply_depends_on_iter)
    {
        if(iteration == 0)
        {
            for(int32_t k = 0; k < num_owned_segments; k++) {
                uint32_t yi  = accu_segment_rows[k];
                uint32_t yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[k];
                auto& c_data = Ct[k];
                Integer_Type v_nitems = v_data.size();
                #pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]], iteration);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
        }
        else
        {
            for(int32_t k = 0; k < num_owned_segments; k++) {
                uint32_t yi  = accu_segment_rows[k];
                uint32_t yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[k];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[k];
                auto& c_data = Ct[k];
                #pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State &state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l], iteration);
                }
            }
        }
    }
    else {
        if(iteration == 0)
        {
            for(int32_t k = 0; k < num_owned_segments; k++) {
                uint32_t yi  = accu_segment_rows[k];
                uint32_t yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                auto& i_data = (*I)[yi];
                auto& iv_data = (*IV)[yi];
                auto& v_data = Vt[k];
                auto& c_data = Ct[k];
                Integer_Type v_nitems = v_data.size();
                #pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = v_data[i];
                    if(i_data[i])
                        c_data[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
                    else
                        c_data[i] = Vertex_Methods.applicator(state);    
                }
            }
        }
        else
        {
            for(int32_t k = 0; k < num_owned_segments; k++) {
                uint32_t yi  = accu_segment_rows[k];
                uint32_t yo = accu_segment_rg;
                std::vector<Fractional_Type>& y_data = Y[yi][yo];
                Integer_Type j = 0;
                auto& iv_data = (*IV)[yi];
                auto& IR = rowgrp_nnz_rows_t[k];
                Integer_Type IR_nitems = IR.size();
                auto& v_data = Vt[k];
                auto& c_data = Ct[k];
                #pragma omp parallel for schedule(static)
                for(Integer_Type j = 0; j < IR_nitems; j++) {
                    Integer_Type i = IR[j];
                    Vertex_State &state = v_data[i];
                    Integer_Type l = iv_data[i];    
                    c_data[i] = Vertex_Methods.applicator(state, y_data[l]);
                }
            }
        }
    }
    /*
    else {
        if(iteration == 0)
        {
            auto& i_data = (*I)[yi];
            auto& iv_data = (*IV)[yi];
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++)
            {
                Vertex_State &state = V[i];
                if(i_data[i])
                    C[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
                else
                    C[i] = Vertex_Methods.applicator(state);    
            }
           
        }
        else
        {
            auto& iv_data = (*IV)[yi];
            auto& IR = rowgrp_nnz_rows_t[0];
            Integer_Type IR_nitems = IR.size();
            //#pragma omp parallel for schedule(static)
            for(Integer_Type k = 0; k < IR_nitems; k++) {
                Integer_Type i =  IR[k];
                Vertex_State &state = V[i];
                //j = iv_data[i];    
                C[i] = Vertex_Methods.applicator(state, y_data[iv_data[i]]);
            }
        }
    }
    */
            

    if(not gather_depends_on_apply and not apply_depends_on_iter) {
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            for(uint32_t j = 0; j < Y[i].size(); j++) {
                std::vector<Fractional_Type> &y_data = Y[i][j];
                Integer_Type y_nitems = y_data.size();
                std::fill(y_data.begin(), y_data.end(), 0);
            }
        }
    }

    if(activity_filtering) {
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            std::vector<char> &t_data = T[i];
            Integer_Type t_nitems = t_data.size();
            std::fill(t_data.begin(), t_data.end(), 0);
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Integer_Type Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::get_vid(Integer_Type index, int32_t segment) {
    return(index + (segment * tile_height));
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
Integer_Type Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::get_vid(Integer_Type index)
{
    //if(tiling_type == _1D_COL_)
    //    return(index + ranks_start_dense[Env::rank]);
    //else
    return(index + (owned_segment * tile_height));
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::
        tile_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile, struct Triple<Weight, Integer_Type> &pair) {
    Integer_Type item1, item2;
    if(ordering_type == _ROW_) {
        item1 = tile.nth;
        item2 = pair.row;
    }
    else if(ordering_type == _COL_) {
        item1 = tile.mth;
        item2 = pair.col;
    }    
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::
        leader_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile) {
    Integer_Type item1, item2;
    if(ordering_type == _ROW_) {
        item1 = tile.leader_rank_rg_rg;
        item2 = Env::rank_rg;
    }
    else if(ordering_type == _COL_) {
        item1 = tile.leader_rank_cg_cg;
        item2 = Env::rank_cg;
    }
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
MPI_Comm Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::communicator_info() {
    MPI_Comm comm = rowgrps_communicator;
    /*
    if(ordering_type == _ROW_) {
        comm = rowgrps_communicator;
    }
    else if(ordering_type == _COL_) {
        comm = rowgrps_communicator;
    }
    */
    return{comm};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::has_converged() {
    
    return(false);
}
    

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::has_converged(int tid) {    
    //bool converged = false;
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;    
    if(check_for_convergence) {
        convergence_vec[tid] = 0;     
        auto& c_data = Ct[tid];
        Integer_Type c_nitems = c_data.size();   
        for(uint32_t i = 0; i < c_nitems; i++) {
            if(not c_data[i]) 
                c_sum_local++;
        }
        if(c_sum_local == c_nitems)
            convergence_vec[tid] = 1;
    }
    
    pthread_barrier_wait(&p_barrier);
    if(tid == 0) {
        iteration++;
        Env::print_num("Iteration", iteration);
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
    
    if(not stationary) {
            
        std::fill(accus_activity_statuses[tid].begin(), accus_activity_statuses[tid].end(), 0);
        
        auto col_chunk_size = msgs_activity_statuses.size() / Env::nthreads;
        auto col_start = msgs_activity_statuses.begin() + (tid * col_chunk_size);
        auto col_end = (tid != Env::nthreads - 1) ? col_start + col_chunk_size : msgs_activity_statuses.end();
        std::fill(col_start, col_end, 0);
    
        /*
        int32_t col_chunk_size = rank_ncolgrps / Env::nthreads;
        int32_t col_start = tid * chunk_size;
        int32_t col_end = (tid != Env::nthreads - 1) ? col_start + col_chunk_size : rank_ncolgrps;
        for(int32_t i = start; i < end; i++) 
            msgs_activity_statuses[i] = 0;   
        }
        */
    }        
    pthread_barrier_wait(&p_barrier);
    
    //Env::barrier();
    //Env::exit(0);
    
    //uint64_t c_sum_local = 0, c_sum_gloabl = 0;
    //if(stationary) {
    /*
    for(int32_t k = 0; k < num_owned_segments; k++) {
        c_sum_local = 0;
        auto& c_data = Ct[k];
        Integer_Type c_nitems = c_data.size();   
        for(uint32_t i = 0; i < c_nitems; i++) {
            if(not c_data[i]) 
                c_sum_local++;
        }
        if(c_sum_local == c_nitems)
            c_sum_local = 1;
        else {
            c_sum_local = 0;
            break;
        }
    }
    */
    /*
    if(check_for_convergence) {
        if(std::accumulate(convergence_vec.begin(), convergence_vec.end(), 0) == Env::nthreads)
            c_sum_local = 1;
        else 
            c_sum_local = 0;
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(c_sum_gloabl == (uint64_t) Env::nranks)
            converged = true;
    } 
    else {
        converged = (iteration >= num_iterations) ? true : false;
    }
      */;
    //MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    //if(c_sum_gloabl == (uint64_t) Env::nranks)
    //    converged = true;
          
    return(converged);   
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::checksum()
{
    uint64_t v_sum_local = 0, v_sum_global = 0;
    //if(stationary) {
        
        for(int32_t k = 0; k < num_owned_segments; k++) {
            auto& v_data = Vt[k];
            Integer_Type v_nitems = v_data.size();
            for(uint32_t i = 0; i < v_nitems; i++) {
                Vertex_State &state = v_data[i];
                if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i, owned_segments[k]) < nrows))    
                        v_sum_local += state.get_state();
                    
            }
        }
    /*    
    } 
    else {
        Integer_Type v_nitems = V.size();
        for(uint32_t i = 0; i < v_nitems; i++) {
            Vertex_State &state = V[i];
            if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i) < nrows))    
                    v_sum_local += state.get_state();
                
        }
    }
    */
    MPI_Allreduce(&v_sum_local, &v_sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master) {
        std::cout << "INFO(rank=" << Env::rank << "): " << "Number of iterations: " << iteration << std::endl;
        std::cout << "INFO(rank=" << Env::rank << "): " << std::fixed << "Value checksum: " << v_sum_global << std::endl;
    }
    

    uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
    //if(stationary) {
        for(int32_t k = 0; k < num_owned_segments; k++) {
            auto& v_data = Vt[k];
            Integer_Type v_nitems = v_data.size();
            for(uint32_t i = 0; i < v_nitems; i++) {
                Vertex_State &state = v_data[i];
                if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i, owned_segments[k]) < nrows)) 
                    v_sum_local_++;
            }
        }
     /*   
    }
    else {
        Integer_Type v_nitems = V.size();
        for(uint32_t i = 0; i < v_nitems; i++) {
            Vertex_State &state = V[i];
            if((state.get_state() != Vertex_Methods.infinity()) and (get_vid(i) < nrows)) 
                v_sum_local_++;
        }
    }
    */
    MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        std::cout << "INFO(rank=" << Env::rank << "): " << std::fixed << "Reachable vertices: " << v_sum_global_ << std::endl;
    Env::barrier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::display(Integer_Type count)
{
    Integer_Type v_nitems;
    if(stationary)
        v_nitems = Vt[0].size();
    else 
        v_nitems = V.size();
    
    count = (v_nitems < count) ? v_nitems : count;
    Env::barrier();
    if(Env::is_master)
    {
        Triple<Weight, Integer_Type> pair, pair1;
        //if(stationary) {
            for(uint32_t i = 0; i < count; i++) {
                pair.row = i;
                pair.col = 0;
                pair1 = A->base(pair, owned_segments[0], owned_segments[0]);
                Vertex_State &state = Vt[0][i];
                std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
            }
            /*
        }
        else {
            for(uint32_t i = 0; i < count; i++){
                pair.row = i;
                pair.col = 0;
                pair1 = A->base(pair, owned_segments[0], owned_segments[0]);
                Vertex_State &state = V[i];
                std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
            }    
        }
        */
            
    }
    Env::barrier();
}

#ifdef TIMING
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State, typename Vertex_Methods_Impl>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State, Vertex_Methods_Impl>::times() {
    if(Env::is_master) {
        double sum = 0.0, mean = 0.0, std_dev = 0.0;
        std::cout << "INFO(rank=" << Env::rank << "): " << "Init    time: " << init_time[0] << " seconds" << std::endl;
        stats(bcast_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Bcast   time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        stats(combine_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Combine time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        stats(apply_time, sum, mean, std_dev);
        std::cout << "INFO(rank=" << Env::rank << "): " << "Apply   time (sum: avg +/- std_dev): " << sum << ": " << mean  << " +/- " << std_dev << " seconds" << std::endl;
        std::cout << "INFO(rank=" << Env::rank << "): " << "Execute time: " << execute_time[0] << " seconds" << std::endl;
        /*
        std::cout << "DETAILED TIMING " << init_time[0] * 1e3;
        stats(bcast_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(combine_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(apply_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        std::cout << " " << execute_time[0] * 1e3 << std::endl;
        */
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
