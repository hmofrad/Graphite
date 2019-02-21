/*
 * vertex.hpp: Vertex program implementation
 * (c) Mohammad Mofrad, 2018
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
class Vertex_Program
{
    public:
        Vertex_Program(Graph<Weight, Integer_Type, Fractional_Type> &Graph,
                        bool stationary_ = false, bool gather_depends_on_apply_ = false, 
                        bool apply_depends_on_iter_ = false, Ordering_type = _ROW_);
        ~Vertex_Program();
        
        virtual bool initializer(Integer_Type vid, Vertex_State &state) { return(stationary);}
        virtual bool initializer(Integer_Type vid, Vertex_State &state, const State &other) { return(stationary);}
        virtual Fractional_Type messenger(Vertex_State &state) { return(1);}
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2, const Fractional_Type &w) { ; }
        virtual void combiner(Fractional_Type &y1, const Fractional_Type &y2) { ; }
        virtual bool applicator(Vertex_State &state, const Fractional_Type &y) { return(true); }
        virtual bool applicator(Vertex_State &state){ return(false); }
        virtual bool applicator(Vertex_State &state, const Fractional_Type &y, const Integer_Type iteration_) { return(true); }
        virtual Fractional_Type infinity() { return(0); }
        virtual bool initializer(Vertex_State &state, const Fractional_Type &v2) { return(stationary);}
        virtual bool initializer(Fractional_Type &v1, const Fractional_Type &v2) { return(stationary);}
        virtual Fractional_Type messenger(Fractional_Type &v, Fractional_Type &s) { return(1);}
        virtual bool applicator(Fractional_Type &v, const Fractional_Type &y) { return(true); }
        virtual bool applicator(Fractional_Type &v, const Fractional_Type &y, Integer_Type iteration_) { return(true); }
        
        void execute(Integer_Type num_iterations_ = 0);
        void initialize();
        template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_>
        void initialize(const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_> &VProgram);
        void free();
        void checksum();
        void display(Integer_Type count = 31);
        
        Integer_Type num_iterations = 0;
        bool stationary = false;
        bool gather_depends_on_apply = false;
        bool apply_depends_on_iter = false;
        Integer_Type iteration = 0;
        std::vector<Vertex_State> V;              // Values
        std::vector<std::vector<Integer_Type>> W; // Values (triangle counting)
    protected:
        bool already_initialized = false;
        bool check_for_convergence = false;
        bool converged = false;
        void init_stationary();
        void init_nonstationary();
        void init_stationary_postprocess();
        void init_nonstationary_postprocess();
        void scatter_gather();
        void scatter_gather_stationary();
        void scatter_gather_nonstationary();
        void scatter_gather_nonstationary_activity_filtering();
        void scatter();
        void gather();
        void bcast();
        void scatter_stationary();
        void gather_stationary();
        void scatter_nonstationary();
        void gather_nonstationary();
        void bcast_stationary();
        void bcast_nonstationary();
        void combine();
        void combine_2d_stationary();
        void combine_2d_nonstationary();
        void combine_postprocess();
        void combine_postprocess_stationary_for_all();
        void combine_postprocess_nonstationary_for_all();
        void apply();                        
        void apply_stationary();
        void apply_nonstationary();
        struct Triple<Weight, double> stats(std::vector<double> &vec);
        void stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev);
        
        void spmv_stationary(struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                std::vector<Fractional_Type> &y_data, 
                std::vector<Fractional_Type> &x_data); // Stationary spmv/spmspv
        void spmv_stationary();
                
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
        Integer_Type get_vid(Integer_Type index);
        
        struct Triple<Weight, Integer_Type> tile_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                       struct Triple<Weight, Integer_Type> &pair);
        struct Triple<Weight, Integer_Type> leader_info( 
                       const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile);  
        MPI_Comm communicator_info();  
                       
        Ordering_type ordering_type;
        Tiling_type tiling_type;
        Compression_type compression_type;
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
        
    
        Matrix<Weight, Integer_Type, Fractional_Type> *A;          // Adjacency list
        /* Stationary */
        std::vector<Fractional_Type> XX;               // Messages 
        std::vector<std::vector<Fractional_Type>> XXT;               // Messages 
        std::vector<Fractional_Type> YY;  // Accumulators
        std::vector<std::vector<Fractional_Type>> YYY;  // Accumulators
        std::vector<std::vector<Fractional_Type>> YYT;  // Accumulators
        std::vector<std::vector<Fractional_Type>> YYYT;  // Accumulators
        
        std::vector<std::vector<char>> IT;
        std::vector<std::vector<Integer_Type>> IVT;
        std::vector<Integer_Type> threads_nnz_rows;
        std::vector<Integer_Type> threads_start_sparse_row;
        
        std::vector<std::vector<int32_t>> threads_send_ranks;
        std::vector<std::vector<int32_t>> threads_send_threads;
        std::vector<std::vector<int32_t>> threads_send_indices;
        std::vector<std::vector<int32_t>> threads_send_tags;
        std::vector<std::vector<Integer_Type>> threads_send_start;
        std::vector<std::vector<Integer_Type>> threads_send_end;
        std::vector<std::vector<int32_t>> threads_recv_ranks;
        std::vector<std::vector<int32_t>> threads_recv_threads;
        std::vector<std::vector<int32_t>> threads_recv_indices;
        std::vector<std::vector<int32_t>> threads_recv_tags;
        std::vector<std::vector<Integer_Type>> threads_recv_start;
        std::vector<std::vector<Integer_Type>> threads_recv_end;
        
        std::vector<std::vector<Integer_Type>> threads_row_start;
        std::vector<std::vector<Integer_Type>> threads_row_end;
        std::vector<std::vector<Integer_Type>> threads_col_start;
        std::vector<std::vector<Integer_Type>> threads_col_end;
        
        
        
        std::vector<Integer_Type> nnz_rows_sizes;
        Integer_Type nnz_rows_size;
        std::vector<Integer_Type> nnz_cols_sizes;
        Integer_Type nnz_cols_size;
        //std::vector<Integer_Type> nnz_cols_size_local;
        
        
        std::vector<std::vector<Fractional_Type>> X;               // Messages 
        std::vector<std::vector<std::vector<Fractional_Type>>> Y;  // Accumulators
        std::vector<char> C;                                       // Convergence vector
        /* Nonstationary */
        std::vector<std::vector<Integer_Type>> XI;                 // X Indices (Nonstationary)
        std::vector<std::vector<Fractional_Type>> XV;              // X Values  (Nonstationary)
        std::vector<std::vector<std::vector<Integer_Type>>> YI;    // Y Indices (Nonstationary)
        std::vector<std::vector<std::vector<Fractional_Type>>> YV; // Y Values (Nonstationary)
        std::vector<std::vector<char>> T;                          // Accumulators activity vectors
        std::vector<Integer_Type> msgs_activity_statuses;
        std::vector<Integer_Type> accus_activity_statuses;
        std::vector<Integer_Type> activity_statuses;
        std::vector<Integer_Type> active_vertices;
        /* Row/Col Filtering indices */
        std::vector<std::vector<char>>* I;
        std::vector<std::vector<Integer_Type>>* IV;
        std::vector<std::vector<char>>* J;
        std::vector<std::vector<Integer_Type>>* JV;
        
        
        std::vector<char>* II;
        std::vector<Integer_Type>* IIV;
        std::vector<char>* JJ;
        std::vector<Integer_Type>* JJV;
        std::vector<Integer_Type> ranks_start_dense;
        std::vector<Integer_Type> ranks_end_dense;   
        std::vector<Integer_Type> ranks_start_sparse;
        std::vector<Integer_Type> ranks_end_sparse;        
        
        
        std::vector<Integer_Type>* rowgrp_nnz_rows;
        std::vector<Integer_Type>* rowgrp_regular_rows;
        std::vector<Integer_Type>* rowgrp_source_rows;
        std::vector<Integer_Type>* colgrp_nnz_columns;
        std::vector<Integer_Type>* colgrp_sink_columns;
        //std::vector<Integer_Type>* rowgrp_REG;
        //std::vector<Integer_Type>* rowgrp_SRC;
        //std::vector<Integer_Type>* rowgrp_SNK;
        //std::vector<Integer_Type>* rowgrp_ZRO;
        //std::vector<Integer_Type>* V2J;
        //std::vector<Integer_Type>* J2V;
        //std::vector<Integer_Type>* Y2V;
        //std::vector<Integer_Type>* V2Y;
        //std::vector<Integer_Type>* I2V;
        //std::vector<Integer_Type>* V2I;

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
        
        #ifdef TIMING
        std::vector<double> init_time;
        std::vector<double> scatter_gather_time;
        std::vector<double> combine_time;
        std::vector<double> apply_time;
        std::vector<double> execute_time;
        #endif
};

/* Support or row-wise tile processing designated to original matrix and 
   column-wise tile processing designated to transpose of the matrix. */                
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::Vertex_Program(
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
    compression_type = A->compression_type;
    //filtering_type = A->filtering_type;
    owned_segment = A->owned_segment;
    leader_ranks = A->leader_ranks;

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
        //accu_segment_row_vec = A->accu_segment_row_vec;
        //accu_segment_col_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_rowgrp_ranks_accu_seg;
        //accu_segment_rg_vec = A->accu_segment_rg_vec;
        accu_segment_rg = A->accu_segment_rg;
        accu_segment_cg = A->accu_segment_cg;
        follower_rowgrp_ranks_rg = A->follower_rowgrp_ranks_rg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_rowgrp_ranks_accu_seg_rg;
        leader_ranks_cg = A->leader_ranks_cg;
        follower_colgrp_ranks_cg = A->follower_colgrp_ranks_cg;
        follower_colgrp_ranks = A->follower_colgrp_ranks;
        local_tiles_row_order = A->local_tiles_row_order;
        local_tiles_col_order = A->local_tiles_col_order;
        local_tiles_row_order_t = A->local_tiles_row_order_t;
        local_tiles_col_order_t = A->local_tiles_col_order_t;
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
        
        II = &(Graph.A->II);
        IIV = &(Graph.A->IIV);
        JJ = &(Graph.A->JJ);
        JJV = &(Graph.A->JJV);
        nnz_rows_sizes = Graph.A->nnz_rows_sizes;
        nnz_cols_sizes = Graph.A->nnz_cols_sizes;
        nnz_rows_size = Graph.A->nnz_rows_size;
        nnz_cols_size = Graph.A->nnz_cols_size;
        ranks_start_dense = Graph.A->ranks_start_dense;
        ranks_end_dense = Graph.A->ranks_end_dense;
        ranks_start_sparse = Graph.A->ranks_start_sparse;
        ranks_end_sparse = Graph.A->ranks_end_sparse;
        
        IT =  Graph.A->IT;
        IVT = Graph.A->IVT;
        threads_nnz_rows = Graph.A->threads_nnz_rows;
        threads_start_sparse_row = Graph.A->threads_start_sparse_row;

        threads_send_ranks = Graph.A->threads_send_ranks;
        threads_send_threads = Graph.A->threads_send_threads;
        threads_send_indices = Graph.A->threads_send_indices;
        threads_send_tags = Graph.A->threads_send_tags;
        threads_send_start = Graph.A->threads_send_start;
        threads_send_end = Graph.A->threads_send_end;        
        threads_recv_ranks = Graph.A->threads_recv_ranks;
        threads_recv_threads = Graph.A->threads_recv_threads;
        threads_recv_indices = Graph.A->threads_recv_indices;
        threads_recv_tags = Graph.A->threads_recv_tags;
        threads_recv_start = Graph.A->threads_recv_start;
        threads_recv_end = Graph.A->threads_recv_end;
        
        out_requests_t.resize(Env::nthreads);
        in_requests_t.resize(Env::nthreads);
        
        threads_row_start = Graph.A->threads_row_start;
        threads_row_end = Graph.A->threads_row_end;
        threads_col_start = Graph.A->threads_col_start;
        threads_col_end = Graph.A->threads_col_end;
        
        
        //rowgrp_REG = &(Graph.A->rowgrp_REG);
        //rowgrp_SRC = &(Graph.A->rowgrp_SRC);
        //rowgrp_SNK = &(Graph.A->rowgrp_SNK);
        //rowgrp_ZRO = &(Graph.A->rowgrp_ZRO);
        //JV = &(Graph.A->JV);
        //V2J = &(Graph.A->V2J);
        //J2V = &(Graph.A->J2V);
        //Y2V = &(Graph.A->Y2V);
        //V2Y = &(Graph.A->V2Y);
        //I2V = &(Graph.A->I2V);
        //V2I = &(Graph.A->V2I);
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
        //accu_segment_col_vec = A->accu_segment_row_vec;
        ///accu_segment_row_vec = A->accu_segment_col_vec;
        all_rowgrp_ranks_accu_seg = A->all_colgrp_ranks_accu_seg;
        //accu_segment_rg_vec = A->accu_segment_cg_vec;
        accu_segment_rg = A->accu_segment_cg;
        accu_segment_cg = A->accu_segment_rg;
        follower_rowgrp_ranks_rg = A->follower_colgrp_ranks_cg;
        follower_rowgrp_ranks_accu_seg_rg = A->follower_colgrp_ranks_accu_seg_cg;
        leader_ranks_cg = A->leader_ranks_rg;
        follower_colgrp_ranks_cg = A->follower_rowgrp_ranks_rg;
        follower_colgrp_ranks = A->follower_rowgrp_ranks;
        local_tiles_row_order = A->local_tiles_col_order;
        local_tiles_col_order = A->local_tiles_row_order;
        local_tiles_row_order_t = A->local_tiles_col_order_t;
        local_tiles_col_order_t = A->local_tiles_row_order_t;
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
        
        II = &(Graph.A->JJ);
        IIV= &(Graph.A->JJV);
        JJ = &(Graph.A->II);
        JJV= &(Graph.A->IIV);
        nnz_rows_sizes = Graph.A->nnz_cols_sizes;
        nnz_cols_sizes = Graph.A->nnz_rows_sizes;
        nnz_rows_size = Graph.A->nnz_cols_size;
        nnz_cols_size = Graph.A->nnz_rows_size;
        ranks_start_dense = Graph.A->ranks_start_dense;
        ranks_end_dense = Graph.A->ranks_end_dense;
        ranks_start_sparse = Graph.A->ranks_start_sparse;
        ranks_end_sparse = Graph.A->ranks_end_sparse;
        
        IT =  Graph.A->IT;
        IVT = Graph.A->IVT;
        threads_nnz_rows = Graph.A->threads_nnz_rows;
        threads_start_sparse_row = Graph.A->threads_start_sparse_row;
        
        threads_send_ranks = Graph.A->threads_send_ranks;
        threads_send_threads = Graph.A->threads_send_threads;
        threads_send_indices = Graph.A->threads_send_indices;
        threads_send_tags = Graph.A->threads_send_tags;
        threads_send_start = Graph.A->threads_send_start;
        threads_send_end = Graph.A->threads_send_end;        
        threads_recv_ranks = Graph.A->threads_recv_ranks;
        threads_recv_threads = Graph.A->threads_recv_threads;
        threads_recv_indices = Graph.A->threads_recv_indices;
        threads_recv_tags = Graph.A->threads_recv_tags;
        threads_recv_start = Graph.A->threads_recv_start;
        threads_recv_end = Graph.A->threads_recv_end;
        
        out_requests_t.resize(Env::nthreads);
        in_requests_t.resize(Env::nthreads);
        
        threads_row_start = Graph.A->threads_col_start;
        threads_row_end = Graph.A->threads_col_end;
        threads_col_start = Graph.A->threads_row_start;
        threads_col_end = Graph.A->threads_row_end;
        
        
        
        //rowgrp_REG = &(Graph.A->rowgrp_REG);
        //rowgrp_SRC = &(Graph.A->rowgrp_SRC);
        //rowgrp_SNK = &(Graph.A->rowgrp_SNK);
        //rowgrp_ZRO = &(Graph.A->rowgrp_ZRO);
        //V2J = &(Graph.A->J2V);
        //J2V = &(Graph.A->V2J);
        //Y2V = &(Graph.A->V2Y);
        //V2Y = &(Graph.A->Y2V);
        //I2V = &(Graph.A->V2I);
        //V2I = &(Graph.A->I2V);
        //if(filtering_type == _SNKS_)
        //    filtering_type = _SRCS_;
        ///else if(filtering_type == _SRCS_)
        //    filtering_type = _SNKS_;
    }   
    
    TYPE_DOUBLE = Types<Weight, Integer_Type, Fractional_Type>::get_data_type();
    TYPE_INT = Types<Weight, Integer_Type, Integer_Type>::get_data_type();
    TYPE_CHAR = Types<Weight, Integer_Type, char>::get_data_type();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::~Vertex_Program() {};

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::free()
{
    V.clear();
    V.shrink_to_fit();

    C.clear();
    C.shrink_to_fit();
    
    if(tiling_type != _1D_COL_) {    
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
    }
    
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::execute(Integer_Type num_iterations_) {
    num_iterations = num_iterations_;
    if(not already_initialized)
        initialize();
    if(!num_iterations)
        check_for_convergence = true; 
    
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    while(true) {
        scatter_gather();
        combine();
        apply();
        iteration++;
        Env::print_num("Iteration: ", iteration);        
        //checksum();
        if(check_for_convergence) {
            converged = has_converged();
            //Env::print_num("Converged: ", converged);            
            if(converged) {
                //combine();
                //apply();
                break;
            }
        }
        else if(iteration >= num_iterations) {
            break;
        }
    }
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Execute", elapsed_time);
    #ifdef TIMING
    execute_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        init_stationary();
        //init_stationary_postprocess();
    }
    else {
        init_stationary();
        init_nonstationary();
        //init_nonstationary_postprocess();
    }
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_State_>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::initialize(
    const Vertex_Program<Weight_, Integer_Type_, Fractional_Type_, Vertex_State_>& VProgram) {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        init_stationary();
        if(tiling_type == _1D_COL_) { 
            auto &i_data = (*II);          
            Integer_Type v_nitems = V.size();
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++) {
                Vertex_State &state = V[i];
                Integer_Type j = i + ranks_start_dense[Env::rank];
                if(i_data[j])
                    C[i] = initializer(get_vid(i), state, (const State&) VProgram.V[i]);
            }            
        }
        else {
            uint32_t yi = accu_segment_row;
            auto &i_data = (*I)[yi];         
            Integer_Type v_nitems = V.size();
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++) {
                Vertex_State &state = V[i];
                if(i_data[i])
                    C[i] = initializer(get_vid(i), state, (const State&) VProgram.V[i]);
            }
        }            
    }
    else {
        init_stationary();
        init_nonstationary();
        //init_nonstationary_postprocess();
    }
    already_initialized = true;
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Init", elapsed_time);
    init_time.push_back(elapsed_time);
    #endif
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_stationary() {
    // Initialize Values
    V.resize(tile_width);
    Integer_Type v_nitems = V.size();
    C.resize(tile_width);
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V[i]; 
        C[i] = initializer(get_vid(i), state);
    }
    // Initialize messages
    if(tiling_type == _1D_COL_) {
        XX.resize(nnz_cols_sizes[Env::rank]);
        //XXT.resize(Env::nthreads);
        //for(int32_t i = 0; i < Env::nthreads; i++)
        //    XXT[i].resize(nnz_cols_sizes[Env::rank]);
        
        //YY.resize(nnz_rows_sizes[Env::rank]);
        YY.resize(nnz_rows_size);
        YYY.resize(rowgrp_nranks - 1);
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
                YYY[j].resize(nnz_rows_sizes[Env::rank]);
        }
        
        //YYT.resize(Env::nthreads);
        //for(int32_t i = 0; i < Env::nthreads; i++) {
        //    YYT[i].resize(threads_nnz_rows[i]);
        //}
    }
    else {    
        // Initialize messages
        std::vector<Integer_Type> x_sizes;
        if(compression_type == _CSC_)
            x_sizes.resize(rank_ncolgrps, tile_height);
        else    
            x_sizes = nnz_col_sizes_loc;
    
        X.resize(rank_ncolgrps);
        for(uint32_t i = 0; i < rank_ncolgrps; i++)
            X[i].resize(x_sizes[i]);
    
        // Initialiaze accumulators
        std::vector<Integer_Type> y_sizes;
        if((compression_type == _CSC_) or (compression_type == _DCSC_))
            y_sizes.resize(rank_nrowgrps, tile_height);
        else    
            y_sizes = nnz_row_sizes_loc;
        
        Y.resize(rank_nrowgrps);
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            if(local_row_segments[i] == owned_segment) {
                Y[i].resize(rowgrp_nranks);
                for(uint32_t j = 0; j < rowgrp_nranks; j++)
                    Y[i][j].resize(y_sizes[i]);
            }
            else {
                Y[i].resize(1);
                Y[i][0].resize(y_sizes[i]);
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::init_nonstationary()
{
    Integer_Type v_nitems = V.size();
    // Initialize activity statuses for all column groups
    // Assuming ncolgrps == nrowgrps
    activity_statuses.resize(ncolgrps);
    
    std::vector<Integer_Type> x_sizes;    
    if(compression_type == _CSC_)
        x_sizes.resize(rank_ncolgrps, tile_height);
    //else if((compression_type == _DCSC_) or (compression_type == _TCSC_))
    else    
        x_sizes = nnz_col_sizes_loc;
    
    // Initialize nonstationary messages values
    XV.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XV[i].resize(x_sizes[i]);
    // Initialize nonstationary messages indices
    XI.resize(rank_ncolgrps);
    for(uint32_t i = 0; i < rank_ncolgrps; i++)
        XI[i].resize(x_sizes[i]);
    
    msgs_activity_statuses.resize(colgrp_nranks);
    
    std::vector<Integer_Type> y_sizes;
    if((compression_type == _CSC_) or (compression_type == _DCSC_))
        y_sizes.resize(rank_nrowgrps, tile_height);
    //else if(compression_type == _TCSC_)    
    else    
        y_sizes = nnz_row_sizes_loc;
    
    T.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
        T[i].resize(y_sizes[i]);
    
    accus_activity_statuses.resize(rowgrp_nranks);
    
    // Initialiaze nonstationary accumulators values
    YV.resize(rank_nrowgrps);
    for(uint32_t i = 0; i < rank_nrowgrps; i++)
    {
        if(local_row_segments[i] == owned_segment)
        {
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
        if(local_row_segments[i] == owned_segment)
        {
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
    
    for(uint32_t k = 0; k < rank_nrowgrps; k++)
    {
        uint32_t yi = k;
        uint32_t yo = 0;
        if(local_row_segments[k] == owned_segment)
            yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        for(uint32_t i = 0; i < y_nitems; i++)
                y_data[i] = infinity();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
        
    if(stationary) {
        scatter_gather_stationary();
        if(not (tiling_type == _1D_COL_) and ((ordering_type == _ROW_ and (tiling_type != _2D_COL_)) 
        or (ordering_type == _COL_ and (tiling_type == _2DT_ or tiling_type == _2D_COL_)))) {
            if(Env::comm_split) {
                if(broadcast_communication)
                    bcast_stationary();
                else {
                    scatter_stationary();
                    gather_stationary();
                }
            }
            else {
                scatter_stationary();
                gather_stationary();
            }
        }
    }
    else {
        scatter_gather_nonstationary();
        if((ordering_type == _ROW_ and tiling_type != _2D_COL_) or (ordering_type == _COL_ and (tiling_type == _2DT_ or tiling_type == _2D_COL_))) {
            if(Env::comm_split) {
                if(broadcast_communication)
                    bcast_nonstationary();
                else {   
                    scatter_nonstationary();
                    gather_nonstationary();
                }
            }
            else {
                scatter_nonstationary();
                gather_nonstationary();
            }
        }
    }
    #ifdef TIMING
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Scatter_gather", elapsed_time);
    scatter_gather_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_stationary() {
    
    if(tiling_type == _1D_COL_) {
        auto& JC = (*colgrp_nnz_columns);
        Integer_Type JC_nitems = JC.size();
        #pragma omp parallel for schedule(static)
        for(uint32_t j = 0; j < JC_nitems; j++) {
            Vertex_State& state = V[JC[j]];
            XX[j] = messenger(state);
        }
        /*
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::copy(XX.begin(), XX.end(), XXT[tid].begin());
        }
        */
    }
    else {    
        uint32_t xo = accu_segment_col;    
        std::vector<Fractional_Type>& x_data = X[xo];
        Integer_Type v_nitems = V.size();
        if(compression_type == _CSC_) {
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < v_nitems; i++) {
                Vertex_State& state = V[i];
                x_data[i] = messenger(state);
            }
        }
        else {    
            auto& JC = (*colgrp_nnz_columns);
            Integer_Type JC_nitems = JC.size();
            #pragma omp parallel for schedule(static)
            for(uint32_t j = 0; j < JC_nitems; j++) {
                Vertex_State& state = V[JC[j]];
                x_data[j] = messenger(state);
            }
        }
    }
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_nonstationary() {
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    std::vector<Fractional_Type> &xv_data = XV[xo];
    std::vector<Integer_Type> &xi_data = XI[xo];
    Integer_Type v_nitems = V.size();
    Integer_Type k = 0;
    if(compression_type == _CSC_) {
        //#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++) {
            Vertex_State &state = V[i];
            if(C[i]) {
                x_data[i] = messenger(state);
                xv_data[k] = x_data[i];
                xi_data[k] = i;
                k++;
            }
            else
                x_data[i] = infinity();
        }        
    }
    else {
        auto& JC = (*colgrp_nnz_columns);
        Integer_Type JC_nitems = JC.size();
        Integer_Type i = 0;
        //#pragma omp parallel for schedule(static)
        for(Integer_Type j = 0; j < JC_nitems; j++) {
            i = JC[j];
            Vertex_State &state = V[i];
            if(C[i]) {
                x_data[j] = messenger(state);    
                xv_data[k] = x_data[j];
                xi_data[k] = j;
                k++;
            }
            else
                x_data[j] = infinity();
        }
    }
    
    if(activity_filtering) {
        msgs_activity_statuses[xo] = k;
        scatter_gather_nonstationary_activity_filtering();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_gather_nonstationary_activity_filtering() {
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


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_stationary() {
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    int32_t col_group = local_col_segments[xo];
    MPI_Request request;
    uint32_t follower;
    for(uint32_t i = 0; i < colgrp_nranks - 1; i++) {
        if(Env::comm_split) {
           follower = follower_colgrp_ranks_cg[i];
           MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
           out_requests.push_back(request);
        }
        else {
            follower = follower_colgrp_ranks[i];
            MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
            out_requests.push_back(request);
        }       
    }
}
template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_stationary() {  
    int32_t leader, my_rank;
    MPI_Request request;
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        std::vector<Fractional_Type> &xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        int32_t col_group = local_col_segments[i];
        if(Env::comm_split) {
            leader = leader_ranks_cg[col_group];
            if(ordering_type == _ROW_)
                my_rank = Env::rank_cg;
            else
                my_rank = Env::rank_rg;
            if(leader != my_rank) {
                MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                in_requests.push_back(request);
            }
        }
        else {
            leader = leader_ranks[col_group];
            if(leader != Env::rank) {
                MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                in_requests.push_back(request);
            }
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_stationary() {
    MPI_Request request;
    int32_t leader;
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        int32_t col_group = local_col_segments[i];
        leader = leader_ranks_cg[col_group];
        std::vector<Fractional_Type> &xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        if(Env::comm_split) {
            MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, colgrps_communicator, &request);
            out_requests.push_back(request);
        }
        else {
            fprintf(stderr, "Invalid communicator\n");
            Env::exit(1);
        }
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();        
} 

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::scatter_nonstationary() {
    uint32_t xo = accu_segment_col;
    std::vector<Fractional_Type> &x_data = X[xo];
    Integer_Type x_nitems = x_data.size();
    std::vector<Fractional_Type> &xv_data = XV[xo];
    std::vector<Integer_Type> &xi_data = XI[xo];
    int nitems = msgs_activity_statuses[xo];
    int32_t col_group = local_col_segments[xo];
    MPI_Request request;
    uint32_t follower;
    for(uint32_t i = 0; i < colgrp_nranks - 1; i++) {
        if(Env::comm_split) {
            follower = follower_colgrp_ranks_cg[i];
            MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, colgrps_communicator);
            if(activity_filtering and nitems) {
                if(nitems > 1) {
                    MPI_Isend(xi_data.data(), nitems - 1, TYPE_INT, follower, col_group, colgrps_communicator, &request);
                    out_requests.push_back(request);
                    MPI_Isend(xv_data.data(), nitems - 1, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
                    out_requests.push_back(request);
                }
            }
            else {
                MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, colgrps_communicator, &request);
                out_requests.push_back(request);
            }
        }
        else {
            follower = follower_colgrp_ranks[i];
            MPI_Send(&nitems, 1, TYPE_INT, follower, col_group, Env::MPI_WORLD);
            if(activity_filtering and nitems) {
                if(nitems > 1) {
                    MPI_Isend(xi_data.data(), nitems - 1, TYPE_INT, follower, col_group, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                    MPI_Isend(xv_data.data(), nitems - 1, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
                    out_requests.push_back(request);
                }
            }
            else {
                MPI_Isend(x_data.data(), x_nitems, TYPE_DOUBLE, follower, col_group, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::gather_nonstationary() {  
    int32_t leader;
    MPI_Request request;
    MPI_Status status;
    int nitems = 0;    
    for(uint32_t i = 0; i < rank_ncolgrps; i++) {
        std::vector<Fractional_Type> &xj_data = X[i];
        Integer_Type xj_nitems = xj_data.size();
        std::vector<Integer_Type> &xij_data = XI[i];
        std::vector<Fractional_Type> &xvj_data = XV[i];
        int32_t col_group = local_col_segments[i];
        if(Env::comm_split) {
            leader = leader_ranks_cg[col_group];
            if(leader != Env::rank_cg) {
                MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, colgrps_communicator, &status);
                msgs_activity_statuses[i] = nitems;
                if(activity_filtering and nitems) {
                    if(nitems > 1) {
                        MPI_Irecv(xij_data.data(), nitems - 1, TYPE_INT, leader, col_group, colgrps_communicator, &request);
                        in_requests.push_back(request);
                        MPI_Irecv(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                        in_requests.push_back(request);
                    }    
                }
                else {
                    MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, colgrps_communicator, &request);
                    in_requests.push_back(request);
                }
            }
        }
        else {
            leader = leader_ranks[col_group];
            if(leader != Env::rank) {
                MPI_Recv(&nitems, 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &status);
                msgs_activity_statuses[i] = nitems;
                if(activity_filtering and nitems) {
                    if(nitems > 1) {
                        MPI_Irecv(xij_data.data(), nitems - 1, TYPE_INT, leader, col_group, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                        MPI_Irecv(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                    }
                }
                else {
                    MPI_Irecv(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader, col_group, Env::MPI_WORLD, &request);
                    in_requests.push_back(request);
                }
            }
        }
    }
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::bcast_nonstationary() {
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
            if(Env::comm_split) {
                if(nitems > 1) {
                    MPI_Ibcast(xij_data.data(), nitems - 1, TYPE_INT, leader_cg, colgrps_communicator, &request);
                    out_requests.push_back(request);
                    MPI_Ibcast(xvj_data.data(), nitems - 1, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
                    out_requests.push_back(request);
                }
            }
            else {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
        }
        else {
            if(Env::comm_split) {
                MPI_Ibcast(xj_data.data(), xj_nitems, TYPE_DOUBLE, leader_cg, colgrps_communicator, &request);
                out_requests.push_back(request);
            }
            else {
                fprintf(stderr, "Invalid communicator\n");
                Env::exit(1);
            }
        }
    }
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();     
}   


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine() {    
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        //if(((not check_for_convergence) and ((iteration + 1) == num_iterations)) or (check_for_convergence and converged))
            
        if((not check_for_convergence) or (check_for_convergence and not converged)) {
            
            if(tiling_type != _1D_COL_) {
                for(uint32_t i = 0; i < rank_nrowgrps; i++) {
                    for(uint32_t j = 0; j < Y[i].size(); j++) {
                        std::vector<Fractional_Type> &y_data = Y[i][j];
                        Integer_Type y_nitems = y_data.size();
                        std::fill(y_data.begin(), y_data.end(), 0);
                    }
                }
            }
            
            combine_2d_stationary();
            combine_postprocess();
        }
        else {
            if(compression_type == _TCSC_) {
                combine_2d_stationary();
                combine_postprocess();
            }
        }
    }
    else {
        if(((not check_for_convergence) or (check_for_convergence and not converged))) {
            combine_2d_nonstationary();
            combine_postprocess();
        }
        //else {
            /*
            if(((not check_for_convergence) or (check_for_convergence and not converged))) {
                combine_2d_nonstationary();
                combine_postprocess();
            }
            else {
                if(compression_type == _TCSC_) {
                    scatter_gather_stationary();
                    bcast_stationary();
                    
                    combine_2d_stationary();
                    combine_postprocess();
                }
            }
            */
                
        //}
                    
        //printf("Still not Converged\n");
        //else
          //  printf("Converged\n");
        



        //printf("combine_postprocess\n\n");
    }
    #ifdef TIMING    
    t2 = Env::clock();
    elapsed_time = t2 - t1;
    Env::print_time("Combine all", elapsed_time);
    combine_time.push_back(elapsed_time);
    #endif
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_stationary() {
    
    //spmv_stationary();
    //Integer_Type offset = 0;
    
    //#pragma omp parallel
    //{
        //int tid = omp_get_thread_num();
    //for(int32_t i = 0; i < Env::nthreads; i++) {
        //std::copy(YYT[tid].begin(), YYT[tid].end(), YY.begin() + threads_start_sparse_row[tid]);
        //offset += YYT[i].size();
    //}
    
   // printf("Here\n");
   // Env::barrier();
    //Env::exit(0);
    //printf("%d %lu %d %f %f\n", offset, YY.size(), nnz_rows_size, YYT[11][YYT[11].size() - 3], YY[YY.size() - 3]);
    
    
    

    /*
    if(tiling_type == _1D_COL_) {
        auto &tile = A->tiles[0][Env::rank];
        spmv_stationary(tile, YY, XX);
        MPI_Request request;
        int32_t leader, follower, tag;
        for(int32_t i = 0; i < Env::nranks; i++) {        
            if(i == Env::rank) {
                int32_t k = 0;
                for(int32_t j = 0; j < Env::nranks; j++) {
                    if(j != Env::rank) {
                        std::vector<Fractional_Type>& yj_data = YYY[k];
                        Integer_Type yj_nitems = yj_data.size();
                        follower = j;
                        tag = i;
                        //printf("%d: recv from %d with size %d %d\n", Env::rank, j, ranks_end_sparse[i] - ranks_start_sparse[i], yj_nitems);
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, tag, Env::MPI_WORLD, &request);
                        in_requests.push_back(request);
                        k++;
                    }
                }
            }
            else {
                //printf("%d: send to %d with size %d\n", Env::rank, i, ranks_end_sparse[i] - ranks_start_sparse[i]);
                leader = i;
                tag = i;
                MPI_Isend(YY.data() + ranks_start_sparse[i], ranks_end_sparse[i] - ranks_start_sparse[i], TYPE_DOUBLE, leader, tag, Env::MPI_WORLD, &request);
                out_requests.push_back(request);
            }
        }
    }
    else if(1){
        */
        //printf("Rank=%d\n", Env::rank);
    #pragma omp parallel
    {   
        int tid = omp_get_thread_num();
        MPI_Request request;
        uint32_t xi= 0, yi = 0, yo = 0;
        for(uint32_t t: local_tiles_row_order) {
            auto pair = A->tile_of_local_tile(t);
            auto &tile = A->tiles[pair.row][pair.col];
            auto pair1 = tile_info(tile, pair); 
            uint32_t tile_th = pair1.row;
            uint32_t pair_idx = pair1.col;
            
            bool vec_owner = leader_ranks[pair_idx] == Env::rank;
            if(vec_owner)
                yo = accu_segment_rg;
            else
                yo = 0;
            
            std::vector<Fractional_Type> &y_data = Y[yi][yo];
            std::vector<Fractional_Type> &x_data = X[xi];
            if(tile.nedges) {
                uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                if(nnz) { 
                    #ifdef HAS_WEIGHT
                    Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                    #endif

                    Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                    Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                    Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
                    
                    if(ordering_type == _ROW_) {
                        for(uint32_t j = 0; j < ncols; j++) {
                            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                #ifdef HAS_WEIGHT
                                combiner(y_data[IA[i]], x_data[j], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[j]);
                                #endif
                            }
                        }
                    }
                    else {
                        for(uint32_t j = 0; j < ncols; j++) {
                            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                #pragma omp critical
                                #ifdef HAS_WEIGHT
                                combiner(y_data[j], x_data[IA[i]], A[i]);   
                                #else
                                combiner(y_data[j], x_data[IA[i]]);
                                #endif
                            }
                        } 
                    }
                }
            }
            
            xi++;
            bool communication = (((tile_th + 1) % rank_ncolgrps) == 0);
            if(communication and not (tiling_type == _2D_ROW_)) {
                MPI_Comm communicator = communicator_info();
                auto pair2 = leader_info(tile);
                int32_t leader = pair2.row;
                int32_t my_rank = pair2.col;
                int32_t follower, accu;
                if(leader == my_rank) {
                    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                        if(Env::comm_split) {   
                            follower = follower_rowgrp_ranks_rg[j];
                            accu = follower_rowgrp_ranks_accu_seg_rg[j];
                        }
                        else {
                            follower = follower_rowgrp_ranks[j];
                            accu = follower_rowgrp_ranks_accu_seg[j];
                        }
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = threads_row_end[yi][tid] - threads_row_start[yi][tid];
                        int32_t tag = (pair_idx * Env::nthreads) + tid;
                        MPI_Irecv(yj_data.data() + threads_row_start[yi][tid], yj_nitems, TYPE_DOUBLE, follower, tag, communicator, &request);
                        in_requests_t[tid].push_back(request);
                        //printf("%d:%d <-- %d:%d [%d %d=%d][%d=%d*%d+%d]\n", Env::rank, tid, follower, tid, threads_send_start[yi][tid], threads_send_end[yi][tid], yj_nitems, tag, pair_idx, Env::nthreads, tid);
                    }
                }
                else {
                    Integer_Type y_nitems = threads_row_end[yi][tid] - threads_row_start[yi][tid];
                    int32_t tag = (pair_idx * Env::nthreads) + tid;
                    MPI_Isend(y_data.data() + threads_row_start[yi][tid], y_nitems, TYPE_DOUBLE, leader, tag, communicator, &request);
                    out_requests_t[tid].push_back(request);
                    //printf("%d:%d --> %d:%d [%d %d=%d][%d=%d*%d+%d]\n", Env::rank, tid, leader, tid, threads_send_start[yi][tid], threads_send_end[yi][tid], y_nitems, tag, pair_idx, Env::nthreads, tid);
                }
                xi = 0;
                yi++;
            }
            //printf("%d %d %d\n", Env::rank, tid, t);
        }
        //printf("|||||||||| %d\n", Env::rank);
        //printf("in_requests_t =  %lu\n", in_requests_t.size());
        Env::barrier();
        MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
        in_requests_t[tid].clear();
        
        uint32_t accu = 0;
        yi = accu_segment_row;
        yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            //Integer_Type yj_nitems = yj_data.size();
            //Integer_Type yj_nitems = threads_send_end[yi][tid] - threads_send_start[yi][tid];
            //#pragma omp parallel for schedule(static)
            for(uint32_t i = threads_row_start[yi][tid]; i < threads_row_end[yi][tid]; i++)
                combiner(y_data[i], yj_data[i]);
        }

        MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
        out_requests_t[tid].clear();

        
    }
    
    printf("RANK=%d\n", Env::rank);
        
        
        
        
        /*
        uint32_t xi= 0, yi = 0;
        for(uint32_t i = 0; i < rank_nrowgrps; i++) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                for(uint32_t j = 0; j < rank_ncolgrps; j++) {  
                    auto pair = A->tile_of_local_tile(local_tiles_row_order_t[i][j]);
                    auto &tile = A->tiles[pair.row][pair.col];
                    auto pair1 = tile_info(tile, pair); 
                    uint32_t tile_th = pair1.row;
                    uint32_t pair_idx = pair1.col;    
                    bool vec_owner = leader_ranks[pair_idx] == Env::rank;
                    uint32_t yo = 0;
                    if(vec_owner)
                        yo = accu_segment_rg;
                    
                    std::vector<Fractional_Type> &y_data = Y[i][yo];
                    std::vector<Fractional_Type> &x_data = X[j];
                    
                    uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
                    if(nnz) { 
                        #ifdef HAS_WEIGHT
                        Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
                        #endif

                        Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
                        Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
                        Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
                        
                        if(ordering_type == _ROW_) {
                            for(uint32_t j = 0; j < ncols; j++) {
                                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                    #ifdef HAS_WEIGHT
                                    combiner(y_data[IA[i]], x_data[j], A[i]);
                                    #else
                                    combiner(y_data[IA[i]], x_data[j]);
                                    #endif
                                }
                            }
                        }
                        else {
                            for(uint32_t j = 0; j < ncols; j++) {
                                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                                    #pragma omp critical
                                    #ifdef HAS_WEIGHT
                                    combiner(y_data[j], x_data[IA[i]], A[i]);   
                                    #else
                                    combiner(y_data[j], x_data[IA[i]]);
                                    #endif
                                }
                            } 
                        }
                    }
                }
                
                //bool communication = (((tile_th + 1) % rank_ncolgrps) == 0);
                //if(communication) {
                MPI_Request request;
                auto pair = A->tile_of_local_tile(local_tiles_row_order_t[i][rank_ncolgrps-1]);
                auto &tile = A->tiles[pair.row][pair.col];
                auto pair1 = tile_info(tile, pair); 
                uint32_t tile_th = pair1.row;
                uint32_t pair_idx = pair1.col;
                uint32_t yo = 0;
                bool vec_owner = leader_ranks[pair_idx] == Env::rank;
                if(vec_owner)
                    yo = accu_segment_rg;
                else
                    yo = 0;
                std::vector<Fractional_Type> &y_data = Y[i][yo];
                
                MPI_Comm communicator = communicator_info();
                auto pair2 = leader_info(tile);
                int32_t leader = pair2.row;
                int32_t my_rank = pair2.col;
                int32_t follower, accu;
                if(leader == my_rank) {
                    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                        if(Env::comm_split) {   
                            follower = follower_rowgrp_ranks_rg[j];
                            accu = follower_rowgrp_ranks_accu_seg_rg[j];
                        }
                        else {
                            follower = follower_rowgrp_ranks[j];
                            accu = follower_rowgrp_ranks_accu_seg[j];
                        }
                        std::vector<Fractional_Type> &yj_data = Y[i][accu];
                        Integer_Type yj_nitems = threads_send_end[i][tid] - threads_send_start[i][tid];
                        int32_t tag = (pair_idx * Env::nthreads) + tid;
                        MPI_Irecv(yj_data.data() + threads_send_start[i][tid], yj_nitems, TYPE_DOUBLE, follower, tag, communicator, &request);
                        in_requests_t[tid].push_back(request);
                        //printf("%d:%d <-- %d [%d %d][%d:%d:%d]\n", Env::rank, tid, follower, threads_send_start[i][tid], threads_send_end[i][tid], yj_nitems, tag, pair_idx);
                    }
                }
                else {
                    Integer_Type y_nitems = threads_send_end[i][tid] - threads_send_start[i][tid];
                    int32_t tag = (pair_idx * Env::nthreads) + tid;
                    MPI_Isend(y_data.data() + threads_send_start[i][tid], y_nitems, TYPE_DOUBLE, leader, tag, communicator, &request);
                    out_requests_t[tid].push_back(request);
                    //printf("%d:%d --> %d [%d %d][%d:%d:%d]\n", Env::rank, tid, leader, threads_send_start[i][tid], threads_send_end[i][tid], y_nitems, tag, pair_idx);
                }
                //}
            }
            
        }
        */
       
    //printf("rank=%d\n", Env::rank);
        
       //Env::barrier();
       //Env::exit(0);        
       /*
    }
    else {
        MPI_Request request;
        uint32_t xi= 0, yi = 0, yo = 0;
        for(uint32_t t: local_tiles_row_order) {
            auto pair = A->tile_of_local_tile(t);
            auto &tile = A->tiles[pair.row][pair.col];
            auto pair1 = tile_info(tile, pair); 
            uint32_t tile_th = pair1.row;
            uint32_t pair_idx = pair1.col;
            
            bool vec_owner = leader_ranks[pair_idx] == Env::rank;
            if(vec_owner)
                yo = accu_segment_rg;
            else
                yo = 0;
            
            std::vector<Fractional_Type> &y_data = Y[yi][yo];
            Integer_Type y_nitems = y_data.size();
            
            std::vector<Fractional_Type> &x_data = X[xi];
            if(tile.nedges)
                spmv_stationary(tile, y_data, x_data);
            
            xi++;
            bool communication = (((tile_th + 1) % rank_ncolgrps) == 0);
            if(communication) {
                MPI_Comm communicator = communicator_info();
                auto pair2 = leader_info(tile);
                int32_t leader = pair2.row;
                int32_t my_rank = pair2.col;
                int32_t follower, accu;
                if(leader == my_rank) {
                    for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {                        
                        if(Env::comm_split) {   
                            follower = follower_rowgrp_ranks_rg[j];
                            accu = follower_rowgrp_ranks_accu_seg_rg[j];
                        }
                        else {
                            follower = follower_rowgrp_ranks[j];
                            accu = follower_rowgrp_ranks_accu_seg[j];
                        }
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = yj_data.size();
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                        in_requests.push_back(request);
                    }
                }
                else {
                    MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                    out_requests.push_back(request);
                }
                xi = 0;
                yi++;
            }
        }
    }
    */
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv_stationary() {
    auto& tile = A->tiles[0][Env::rank];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& y_data = YYT[tid];
        auto& x_data = XXT[tid];
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
                    combiner(y_data[IA[i]], x_data[j], A[i]);
                    #else
                    combiner(y_data[IA[i]], x_data[j]);
                    #endif
                    //printf("%f\n", x_data[j]);
                }
            }
            std::copy(YYT[tid].begin(), YYT[tid].end(), YY.begin() + threads_start_sparse_row[tid]);
        }
    }
}



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv_stationary(
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<Fractional_Type> &y_data,
            std::vector<Fractional_Type> &x_data) {
                


    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t nnz = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnz;
        if(nnz) {
            #ifdef HAS_WEIGHT
            Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
            #endif

            Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
            Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
            Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
            
            if(ordering_type == _ROW_) {
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #ifdef HAS_WEIGHT
                        combiner(y_data[IA[i]], x_data[j], A[i]);
                        #else
                        combiner(y_data[IA[i]], x_data[j]);
                        #endif
                    }
                }
            }
            else {
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #pragma omp critical
                        #ifdef HAS_WEIGHT
                        combiner(y_data[j], x_data[IA[i]], A[i]);   
                        #else
                        combiner(y_data[j], x_data[IA[i]]);
                        #endif
                    }
                } 
            }
        }
        

        
        
  
        
        
        
        
  
        /*
        MPI_Request request;
        int32_t recv_count = threads_recv_ranks[tid].size();
        for(int32_t i = 0; i < (int32_t) threads_recv_ranks[tid].size(); i++) {
            MPI_Irecv(YYY[threads_recv_indices[tid][i]].data() + (threads_recv_start[tid][i] - ranks_start_sparse[Env::rank]), threads_recv_end[tid][i] - threads_recv_start[tid][i], TYPE_DOUBLE, threads_recv_ranks[tid][i], threads_recv_tags[tid][i], Env::MPI_WORLD, &request);
            //#pragma omp critical
            //in_requests.push_back(request);
            in_requests_t[tid].push_back(request);
            //if(Env::rank == -1)
                //printf("Recv: %d:%d --> %d:%d [%d %d] %d [%d %d] %d %d\n", Env::rank, tid, threads_recv_ranks[tid][i], threads_recv_threads[tid][i], threads_recv_start[tid][i], threads_recv_end[tid][i], threads_recv_tags[tid][i], threads_recv_start[tid][i] - ranks_start_sparse[Env::rank], threads_recv_end[tid][i] - ranks_start_sparse[Env::rank], threads_recv_end[tid][i] - threads_recv_start[tid][i], threads_recv_indices[tid][i]);
        }
        
        int32_t send_count = threads_send_ranks[tid].size();
        for(int32_t i = 0; i < send_count; i++) {
            MPI_Isend(YY.data() + threads_send_start[tid][i], threads_send_end[tid][i] - threads_send_start[tid][i], TYPE_DOUBLE, threads_send_ranks[tid][i], threads_send_tags[tid][i], Env::MPI_WORLD, &request);
            //#pragma omp critical
            //out_requests.push_back(request);
            out_requests_t[tid].push_back(request);
            //if(Env::rank == -1)
                //printf("Send: %d:%d --> %d:%d [%d %d] %d\n", Env::rank, tid, threads_send_ranks[tid][i], threads_send_threads[tid][i], threads_send_start[tid][i], threads_send_end[tid][i], threads_send_tags[tid][i]);
        }
        */
        

        
        
        
        
    }
    //printf("rank=%d\n", Env::rank);
    //Env::barrier();
    //Env::exit(0);
    
    //std::exit(0);
    
    
    /*
    if(compression_type == _CSC_) {
        #ifdef HAS_WEIGHT
        A = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->ncols;
    }
    else if(compression_type == _DCSC_) {
        #ifdef HAS_WEIGHT
        A = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;   
        JC   = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JC;           
        ncols = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
    }
    else if(compression_type == _TCSC_) {
        #ifdef HAS_WEIGHT
        A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;                
    }
    else {    
        #ifdef HAS_WEIGHT
        A = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
    }
    
    
    if((compression_type == _CSC_) or (compression_type == _TCSC_)) {        
        if(ordering_type == _ROW_) {
            for(uint32_t j = 0; j < ncols; j++) {
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                    #ifdef HAS_WEIGHT
                    combiner(y_data[IA[i]], x_data[j], A[i]);
                    #else
                    combiner(y_data[IA[i]], x_data[j]);
                    #endif
                }
            }
        }
        else {
            for(uint32_t j = 0; j < ncols; j++) {
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                    #ifdef HAS_WEIGHT
                    combiner(y_data[j], x_data[IA[i]], A[i]);   
                    #else
                    combiner(y_data[j], x_data[IA[i]]);
                    #endif
                }
            }       
        }
    }
    else if(compression_type == _DCSC_) {
        if(ordering_type == _ROW_) {
            auto& iv_data = (*IV)[tile.jth];
            for(uint32_t j = 0; j < ncols; j++) {
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                    #ifdef HAS_WEIGHT
                    combiner(y_data[IA[i]], x_data[j], A[i]);
                    #else
                    combiner(y_data[IA[i]], x_data[j]);
                    #endif
                }
            }
        }
        else {

            Integer_Type k = 0;
            Integer_Type l = 0;
            auto& jv_data = (*JV)[tile.ith];
            for(uint32_t j = 0; j < ncols; j++) {
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                    k = jv_data[IA[i]];
                    l = JC[j];
                    #ifdef HAS_WEIGHT
                    combiner(y_data[l], x_data[k], A[i]);   
                    #else
                    combiner(y_data[l], x_data[k]);
                    #endif
                }
            }            
        }
    }
    else {
        if(num_iterations == 1) {
            if(ordering_type == _ROW_) {
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #ifdef HAS_WEIGHT
                        combiner(y_data[IA[i]], x_data[j], A[i]);
                        #else
                        combiner(y_data[IA[i]], x_data[j]);
                        #endif
                    }
                } 
            }
            else { 
                for(uint32_t j = 0; j < ncols; j++) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #ifdef HAS_WEIGHT
                        combiner(y_data[j], x_data[IA[i]], A[i]);   
                        #else
                        combiner(y_data[j], x_data[IA[i]]);
                        #endif
                    }
                }
            }            
        }
        else {
            if(ordering_type == _ROW_) {                
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
                                combiner(y_data[IA[i]], x_data[l], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[l]);
                                #endif
                            }
                        }                    
                    }
                }
                
                if((not check_for_convergence) or (check_for_convergence and not converged)) {
                    //printf("1.iter=%d\n", iteration);
                    Integer_Type NC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_REG_R_REG_C;
                    if(NC_REG_R_REG_C) {
                        Integer_Type* JC_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_REG_R_REG_C;
                        Integer_Type* JA_REG_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_REG_R_REG_C;
                        for(uint32_t j = 0, k = 0; j < NC_REG_R_REG_C; j++, k = k + 2) {
                            l = JC_REG_R_REG_C[j];
                            for(uint32_t i = JA_REG_R_REG_C[k]; i < JA_REG_R_REG_C[k + 1]; i++) {
                                #ifdef HAS_WEIGHT
                                combiner(y_data[IA[i]], x_data[l], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[l]);
                                #endif
                            }
                        }                    
                    } 
                }                
                if(((not check_for_convergence) and ((iteration + 1) == num_iterations)) or (check_for_convergence and converged)) {
                    //printf("2.iter=%d\n", iteration);
                    //if(stationary) {
                    Integer_Type NC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->NC_SRC_R_REG_C;
                    if(NC_SRC_R_REG_C) {
                        Integer_Type* JC_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JC_SRC_R_REG_C;
                        Integer_Type* JA_SRC_R_REG_C = static_cast<TCSC_CF_BASE<Weight, Integer_Type>*>(tile.compressor)->JA_SRC_R_REG_C;
                        for(uint32_t j = 0, k = 0; j < NC_SRC_R_REG_C; j++, k = k + 2) {
                            l = JC_SRC_R_REG_C[j];
                            for(uint32_t i = JA_SRC_R_REG_C[k]; i < JA_SRC_R_REG_C[k + 1]; i++) {
                                #ifdef HAS_WEIGHT
                                combiner(y_data[IA[i]], x_data[l], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[l]);
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
                                combiner(y_data[IA[i]], x_data[l], A[i]);
                                #else
                                combiner(y_data[IA[i]], x_data[l]);
                                #endif
                            }
                        }                    
                    }
                }             
            }
            else {
                fprintf(stderr, "Not implemented\n");
                Env::exit(0);
            }                
                
        }
    }
    */
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_2d_nonstationary() {
    #ifdef TIMING
    double t1, t2, elapsed_time = 0;
    #endif
    MPI_Request request;
    uint32_t tile_th, pair_idx;
    int32_t leader, follower, my_rank, accu;
    bool vec_owner, communication;
    uint32_t xi= 0, yi = 0, yo = 0;
    for(uint32_t t: local_tiles_row_order) {
        auto pair = A->tile_of_local_tile(t);
        auto &tile = A->tiles[pair.row][pair.col];
        auto pair1 = tile_info(tile, pair); 
        tile_th = pair1.row;
        pair_idx = pair1.col;
        vec_owner = leader_ranks[pair_idx] == Env::rank;
        if(vec_owner)
            yo = accu_segment_rg;
        else
            yo = 0;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type y_nitems = y_data.size();
        std::vector<Fractional_Type> &x_data = X[xi];
        std::vector<Fractional_Type> &xv_data = XV[xi];
        std::vector<Integer_Type> &xi_data = XI[xi];
        std::vector<char> &t_data = T[yi];
        if(tile.nedges)
            spmv_nonstationary(tile, y_data, x_data, xv_data, xi_data, t_data);
        
        xi++;
        communication = (((tile_th + 1) % rank_ncolgrps) == 0);
        if(communication) {
            MPI_Comm communicator = communicator_info();
            auto pair2 = leader_info(tile);
            leader = pair2.row;
            my_rank = pair2.col;
            if(leader == my_rank) {
                for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
                    if(Env::comm_split) {   
                        follower = follower_rowgrp_ranks_rg[j];
                        accu = follower_rowgrp_ranks_accu_seg_rg[j];
                    }
                    else {
                        follower = follower_rowgrp_ranks[j];
                        accu = follower_rowgrp_ranks_accu_seg[j];
                    }
                    if(activity_filtering and activity_statuses[tile.rg]) {
                        // 0 all / 1 nothing / else nitems 
                        int nitems = 0;
                        MPI_Status status;
                        MPI_Recv(&nitems, 1, MPI_INT, follower, pair_idx, communicator, &status);
                        accus_activity_statuses[accu] = nitems;
                        if(accus_activity_statuses[accu] > 1) {
                            std::vector<Integer_Type> &yij_data = YI[yi][accu];
                            std::vector<Fractional_Type> &yvj_data = YV[yi][accu];
                            MPI_Irecv(yij_data.data(), accus_activity_statuses[accu] - 1, TYPE_INT, follower, pair_idx, communicator, &request);
                            in_requests.push_back(request);
                            MPI_Irecv(yvj_data.data(), accus_activity_statuses[accu] - 1, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                            in_requests_.push_back(request);
                        }
                    }
                    else {                                
                        std::vector<Fractional_Type> &yj_data = Y[yi][accu];
                        Integer_Type yj_nitems = yj_data.size();
                        MPI_Irecv(yj_data.data(), yj_nitems, TYPE_DOUBLE, follower, pair_idx, communicator, &request);
                        in_requests.push_back(request);
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
                    MPI_Send(&nitems, 1, TYPE_INT, leader, pair_idx, communicator);
                    if(nitems > 1) {
                        MPI_Isend(yi_data.data(), nitems - 1, TYPE_INT, leader, pair_idx, communicator, &request);
                        out_requests.push_back(request);
                        MPI_Isend(yv_data.data(), nitems - 1, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                        out_requests_.push_back(request);
                    }
                }
                else {
                    MPI_Isend(y_data.data(), y_nitems, TYPE_DOUBLE, leader, pair_idx, communicator, &request);
                    out_requests.push_back(request);
                }
            }
            xi = 0;
            yi++;
        }
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::spmv_nonstationary(
            struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
            std::vector<Fractional_Type> &y_data,
            std::vector<Fractional_Type> &x_data,
            std::vector<Fractional_Type> &xv_data, 
            std::vector<Integer_Type> &xi_data, std::vector<char> &t_data) {
     
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #ifdef HAS_WEIGHT
        Integer_Type* A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->A;
        #endif

        Integer_Type* IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->IA;
        Integer_Type* JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->JA;    
        Integer_Type ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor_t[tid])->nnzcols;  
        
        
        if(activity_filtering and activity_statuses[tile.cg]) {
            Integer_Type s_nitems = msgs_activity_statuses[tile.jth] - 1;
            Integer_Type j = 0;
            for(Integer_Type k = 0; k < s_nitems; k++) {
                j = xi_data[k];
                for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                    #ifdef HAS_WEIGHT
                    combiner(y_data[IA[i]], xv_data[k], A[i]);
                    #else
                    combiner(y_data[IA[i]], xv_data[k]);
                    #endif
                    t_data[IA[i]] = 1;
                }
            }
        }
        else {
            for(uint32_t j = 0; j < ncols; j++) {
                if(x_data[j] != infinity()) {
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #ifdef HAS_WEIGHT
                        combiner(y_data[IA[i]], x_data[j], A[i]);
                        #else
                        combiner(y_data[IA[i]], x_data[j]);
                        #endif
                        t_data[IA[i]] = 1;
                    }
                }
            }
        }
    }     
                
                
         /*       
    #ifdef HAS_WEIGHT
    Weight* A;
    #endif
    Integer_Type* IA;
    Integer_Type* JA;
    Integer_Type ncols;
    if(compression_type == _CSC_) {
        #ifdef HAS_WEIGHT
        A = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<CSC_BASE<Weight, Integer_Type>*>(tile.compressor)->ncols;
    }
    else if(compression_type == _DCSC_) {
        #ifdef HAS_WEIGHT
        A = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<DCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
    }
    else {    
        #ifdef HAS_WEIGHT
        A = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->A;
        #endif
        IA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->IA;
        JA   = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->JA;    
        ncols = static_cast<TCSC_BASE<Weight, Integer_Type>*>(tile.compressor)->nnzcols;        
    }

            if(activity_filtering and activity_statuses[tile.cg]) {
                Integer_Type s_nitems = msgs_activity_statuses[tile.jth] - 1;
                Integer_Type j = 0;
                for(Integer_Type k = 0; k < s_nitems; k++) {
                    j = xi_data[k];
                    for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                        #ifdef HAS_WEIGHT
                        combiner(y_data[IA[i]], xv_data[k], A[i]);
                        #else
                        combiner(y_data[IA[i]], xv_data[k]);
                        #endif
                        t_data[IA[i]] = 1;
                    }
                }
            }
            else {
                for(uint32_t j = 0; j < ncols; j++) {
                    if(x_data[j] != infinity()) {
                        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                            #ifdef HAS_WEIGHT
                            combiner(y_data[IA[i]], x_data[j], A[i]);
                            #else
                            combiner(y_data[IA[i]], x_data[j]);
                            #endif
                            t_data[IA[i]] = 1;
                        }
                    }
                }
            }
            */
        //}
    //}
}



template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess() {
    if(stationary)
        combine_postprocess_stationary_for_all();
    else {
        combine_postprocess_nonstationary_for_all();
        
        std::fill(msgs_activity_statuses.begin(), msgs_activity_statuses.end(), 0);
        std::fill(accus_activity_statuses.begin(), accus_activity_statuses.end(), 0);
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_stationary_for_all()
{
    if(tiling_type == _1D_COL_) {
        /*
        for(int i = 0; i < Env::nthreads; i++) {
            printf("%d %d %d %d\n", Env::rank, i, in_requests_t[i].size(), out_requests_t[i].size());
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
            in_requests_t[tid].clear();
            
            MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
            out_requests_t[tid].clear();
        }
        
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            std::vector<Fractional_Type>& yj_data = YYY[j];
            Integer_Type yj_nitems = yj_data.size();
            
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++) {
                Integer_Type k = i + ranks_start_sparse[Env::rank];
                combiner(YY[k], yj_data[i]);
            }
        }
        */
        
        
        
        wait_for_recvs();  
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            std::vector<Fractional_Type>& yj_data = YYY[j];
            Integer_Type yj_nitems = yj_data.size();
            
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++) {
                Integer_Type k = i + ranks_start_sparse[Env::rank];
                combiner(YY[k], yj_data[i]);
            }
        }
        wait_for_sends();
        
    }
    else {
        ;
        /*
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            MPI_Waitall(in_requests_t[tid].size(), in_requests_t[tid].data(), MPI_STATUSES_IGNORE);
            in_requests_t[tid].clear();
        }
        */
        /*
        for(int i = 0; i < Env::nthreads; i++) {
            MPI_Waitall(in_requests_t[i].size(), in_requests_t[i].data(), MPI_STATUSES_IGNORE);
            in_requests_t[i].clear();
        }
        */
        
        //wait_for_recvs();
        /*
        uint32_t accu = 0;
        uint32_t yi = accu_segment_row;
        uint32_t yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {
            if(Env::comm_split)
                accu = follower_rowgrp_ranks_accu_seg_rg[j];
            else
                accu = follower_rowgrp_ranks_accu_seg[j];
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                combiner(y_data[i], yj_data[i]);
        }
        */
        //wait_for_sends();
        /*
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            MPI_Waitall(out_requests_t[tid].size(), out_requests_t[tid].data(), MPI_STATUSES_IGNORE);
            out_requests_t[tid].clear();
        }
        */
        /*
        for(int i = 0; i < Env::nthreads; i++) {
            MPI_Waitall(out_requests_t[i].size(), out_requests_t[i].data(), MPI_STATUSES_IGNORE);
            out_requests_t[i].clear();
        }
        */
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::combine_postprocess_nonstationary_for_all() {
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
                #pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < accus_activity_statuses[accu] - 1; i++) {
                    Integer_Type k = yij_data[i];
                    combiner(y_data[k], yvj_data[i]);
                }
            }
        }
        else {
            std::vector<Fractional_Type> &yj_data = Y[yi][accu];
            Integer_Type yj_nitems = yj_data.size();
            #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i < yj_nitems; i++)
                combiner(y_data[i], yj_data[i]);
        }
    }
    wait_for_sends();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_all() {
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_sends() {
    MPI_Waitall(out_requests.size(), out_requests.data(), MPI_STATUSES_IGNORE);
    out_requests.clear();
    if(not stationary and activity_filtering) {
        MPI_Waitall(out_requests_.size(), out_requests_.data(), MPI_STATUSES_IGNORE);
        out_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::wait_for_recvs() {
    MPI_Waitall(in_requests.size(), in_requests.data(), MPI_STATUSES_IGNORE);
    in_requests.clear();
    if(not stationary and activity_filtering) {
        MPI_Waitall(in_requests_.size(), in_requests_.data(), MPI_STATUSES_IGNORE);
        in_requests_.clear();
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply() {
    #ifdef TIMING
    double t1, t2, elapsed_time;
    t1 = Env::clock();
    #endif
    if(stationary) {
        if(not converged)
            apply_stationary();
        else {
            if(compression_type == _TCSC_)
                apply_stationary();
        }
    }
    else {
        if(not converged)
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_stationary() {
    if(tiling_type == _1D_COL_) {        
        auto& i_data = (*II);
        auto& iv_data = (*IIV);
        Integer_Type v_nitems = V.size();
        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < v_nitems; i++)
        {
            Vertex_State& state = V[i];
            Integer_Type j = i + ranks_start_dense[Env::rank];
            if(i_data[j]) {
                C[i] = applicator(state, YY[iv_data[j]]);
            }
            else
                C[i] = applicator(state);
        }    
        std::fill(YY.begin(), YY.end(), 0);
        for(uint32_t j = 0; j < rowgrp_nranks - 1; j++) {        
            std::fill(YYY[j].begin(), YYY[j].end(), 0);
        }
        //for(int32_t i = 0; i < Env::nthreads; i++) {
        //    std::fill(YYT[i].begin(), YYT[i].end(), 0);
        //}
    }
    else {
        uint32_t yi = accu_segment_row;
        uint32_t yo = accu_segment_rg;
        std::vector<Fractional_Type> &y_data = Y[yi][yo];
        Integer_Type v_nitems = V.size();
        if((compression_type == _CSC_) or (compression_type == _DCSC_)){
            //if(not converged) {
                #pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    C[i] = applicator(state, y_data[i]);
                }        
            //}
        }
        else if (compression_type == _TCSC_){
            //if(not converged) {
                auto& i_data = (*I)[yi];
                auto &iv_data = (*IV)[yi];
                //Integer_Type j = 0;
                #pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i]) {
                        //C[i] = applicator(state, y_data[j]);
                        //j++;
                        C[i] = applicator(state, y_data[iv_data[i]]);
                    }
                    else
                        C[i] = applicator(state);
                }        
            //}
        }
        else {
            Integer_Type j = 0;
            auto& iv_data = (*IV)[yi];
            
            if((not check_for_convergence) or (check_for_convergence and not converged)) {
                auto& regular_rows = (*rowgrp_regular_rows);            
                for(Integer_Type i: regular_rows) {
                    Vertex_State &state = V[i];
                    j = iv_data[i];    
                    C[i] = applicator(state, y_data[j]);
                }
            }
            if(((not check_for_convergence) and ((iteration + 1) == num_iterations)) or (check_for_convergence and converged)) {
                auto& source_rows = (*rowgrp_source_rows);
                j = 0;
                for(Integer_Type i: source_rows) {
                    Vertex_State &state = V[i];
                    j = iv_data[i];
                    C[i] = applicator(state, y_data[j]);
                }
            }
        }
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::apply_nonstationary()
{
    uint32_t yi = accu_segment_row;
    uint32_t yo = accu_segment_rg;
    std::vector<Fractional_Type> &y_data = Y[yi][yo];

    Integer_Type v_nitems = V.size();
    
    if((compression_type == _CSC_) or (compression_type == _DCSC_)) {
        //if(not converged) {
            if(apply_depends_on_iter) {
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++) {
                    Vertex_State &state = V[i];
                    C[i] = applicator(state, y_data[i], iteration);
                }
            }
            else {
                //#pragma omp parallel for schedule(static)
                for(uint32_t i = 0; i < v_nitems; i++) {
                    Vertex_State &state = V[i];
                    C[i] = applicator(state, y_data[i]);
                }
            }
        //}
    }
    //else if(compression_type == _TCSC1_) {
    else
    {
        //if(not converged) {
            if(apply_depends_on_iter)
            {
                if(iteration == 0)
                {
                    auto &i_data = (*I)[yi];
                    auto &iv_data = (*IV)[yi];
                    #pragma omp parallel for schedule(static)
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        Vertex_State &state = V[i];
                        if(i_data[i])
                            C[i] = applicator(state, y_data[iv_data[i]], iteration);
                        else
                            C[i] = applicator(state);    
                    }
                }
                else
                {
                    Integer_Type j = 0;
                    auto& iv_data = (*IV)[yi];
                    auto& IR = (*rowgrp_nnz_rows);
                    Integer_Type IR_nitems = IR.size();
                    //Integer_Type i = 0;
                    #pragma omp parallel for schedule(static)
                    //for(Integer_Type i: IR) {
                    for(Integer_Type k = 0; k < IR_nitems; k++) {
                        Integer_Type i =  IR[k];
                        Vertex_State &state = V[i];
                        j = iv_data[i];    
                        C[i] = applicator(state, y_data[j], iteration);
                    }

                    /*
                    auto& iv_data = (*IV)[yi];
                    Integer_Type j = 0;
                    auto& regular_rows = (*rowgrp_regular_rows);
                    for(Integer_Type i: regular_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];    
                        C[i] = applicator(state, y_data[j], iteration);
                    }

                    auto& source_rows = (*rowgrp_source_rows);
                    j = 0;
                    for(Integer_Type i: source_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];
                        C[i] = applicator(state, y_data[j], iteration);
                    }
                    */
                }
            }
            else {
                if(iteration == 0)
                {
                    auto &i_data = (*I)[yi];
                    auto &iv_data = (*IV)[yi];
                    #pragma omp parallel for schedule(static)
                    for(uint32_t i = 0; i < v_nitems; i++)
                    {
                        Vertex_State &state = V[i];
                        if(i_data[i])
                            C[i] = applicator(state, y_data[iv_data[i]]);
                        else
                            C[i] = applicator(state);    
                    }
                   
                }
                else
                {
                    //Integer_Type j = 0;
                    auto& iv_data = (*IV)[yi];
                    auto& IR = (*rowgrp_nnz_rows);
                    Integer_Type IR_nitems = IR.size();
                    //Integer_Type i = 0;
                    #pragma omp parallel for schedule(static)
                    //for(Integer_Type i: IR) {
                    for(Integer_Type k = 0; k < IR_nitems; k++) {
                        Integer_Type i =  IR[k];
                        Vertex_State &state = V[i];
                        //j = iv_data[i];    
                        C[i] = applicator(state, y_data[iv_data[i]]);
                    }
                    
                    /*
                    auto& iv_data = (*IV)[yi];
                    Integer_Type j = 0;

                    auto& regular_rows = (*rowgrp_regular_rows);
                    for(Integer_Type i: regular_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];    
                        C[i] = applicator(state, y_data[j]);
                    }

                    auto& source_rows = (*rowgrp_source_rows);
                    j = 0;
                    for(Integer_Type i: source_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];
                        C[i] = applicator(state, y_data[j]);
                    }
                    */
                }
            }
        //}            
    }
    /*
    else {
        if(apply_depends_on_iter)
        {
            if(iteration == 0)
            {
                auto &i_data = (*I)[yi];
                auto &iv_data = (*IV)[yi];
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                        C[i] = applicator(state, y_data[iv_data[i]], iteration);
                    else
                        C[i] = applicator(state);    
                }
            }
            else
            {
                
                //Integer_Type j = 0;
                //auto& iv_data = (*IV)[yi];
                //auto& IR = (*rowgrp_nnz_rows);
                //Integer_Type IR_nitems = IR.size();
                //Integer_Type i = 0;
                //for(Integer_Type i: IR) {
                //    Vertex_State &state = V[i];
                //    j = iv_data[i];    
                //    C[i] = applicator(state, y_data[j], iteration);
                //}
                
                auto& iv_data = (*IV)[yi];
                Integer_Type j = 0;
                if((not check_for_convergence) or (check_for_convergence and not converged)) {
                    auto& regular_rows = (*rowgrp_regular_rows);            
                    for(Integer_Type i: regular_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];    
                        C[i] = applicator(state, y_data[j], iteration);
                    }
                }
                if(((not check_for_convergence) and ((iteration + 1) == num_iterations)) or (check_for_convergence and converged)) {
                    auto& source_rows = (*rowgrp_source_rows);
                    j = 0;
                    for(Integer_Type i: source_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];
                        C[i] = applicator(state, y_data[j], iteration);
                    }
                }
            }
        }
        else {
            if(iteration == 0)
            {
                auto &i_data = (*I)[yi];
                auto &iv_data = (*IV)[yi];
                for(uint32_t i = 0; i < v_nitems; i++)
                {
                    Vertex_State &state = V[i];
                    if(i_data[i])
                        C[i] = applicator(state, y_data[iv_data[i]]);
                    else
                        C[i] = applicator(state);    
                }
               
            }
            else
            {
                
                //Integer_Type j = 0;
                //auto& iv_data = (*IV)[yi];
                //auto& IR = (*rowgrp_nnz_rows);
                //Integer_Type IR_nitems = IR.size();
                //Integer_Type i = 0;
                //for(Integer_Type i: IR) {
                //    Vertex_State &state = V[i];
                //    j = iv_data[i];    
                //    C[i] = applicator(state, y_data[j]);
                //}
                
                
                auto& iv_data = (*IV)[yi];
                Integer_Type j = 0;
                if((not check_for_convergence) or (check_for_convergence and not converged)) {
                    auto& regular_rows = (*rowgrp_regular_rows);            
                    for(Integer_Type i: regular_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];    
                        C[i] = applicator(state, y_data[j]);
                    }
                }
                if(((not check_for_convergence) and ((iteration + 1) == num_iterations)) or (check_for_convergence and converged)) {
                    auto& source_rows = (*rowgrp_source_rows);
                    j = 0;
                    for(Integer_Type i: source_rows) {
                        Vertex_State &state = V[i];
                        j = iv_data[i];
                        C[i] = applicator(state, y_data[j]);
                    }
                }
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

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
Integer_Type Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::get_vid(Integer_Type index)
{
    if(tiling_type == _1D_COL_)
        return(index + ranks_start_dense[Env::rank]);
    else
        return(index + (owned_segment * tile_height));
    
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::
        tile_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile,
                  struct Triple<Weight, Integer_Type> &pair)
{
    Integer_Type item1, item2;
    if(ordering_type == _ROW_)
    {
        item1 = tile.nth;
        item2 = pair.row;
    }
    else if(ordering_type == _COL_)
    {
        item1 = tile.mth;
        item2 = pair.col;
    }    
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, Integer_Type> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::
        leader_info(const struct Tile2D<Weight, Integer_Type, Fractional_Type> &tile)
{
    Integer_Type item1, item2;
    if(ordering_type == _ROW_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_rg_rg;
            item2 = Env::rank_rg;
        }
        else
        {
            item1 = tile.leader_rank_rg;
            item2 = Env::rank;
        }
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
        {
            item1 = tile.leader_rank_cg_cg;
            item2 = Env::rank_cg;
        }
        else
        {
            item1 = tile.leader_rank_cg;
            item2 = Env::rank;
        }
    }
    return{item1, item2};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
MPI_Comm Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::communicator_info()
{
    MPI_Comm comm;
    if(ordering_type == _ROW_)
    {
        if(Env::comm_split)
            comm = rowgrps_communicator;
        else
            comm = Env::MPI_WORLD;
    }
    else if(ordering_type == _COL_)
    {
        if(Env::comm_split)
            comm = rowgrps_communicator;
        else
            comm = Env::MPI_WORLD;
    }
    return{comm};
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
bool Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::has_converged()
{
    bool converged = false;
    uint64_t c_sum_local = 0, c_sum_gloabl = 0;
        if((compression_type == _CSC_) or (compression_type == _DCSC_) or (compression_type == _TCSC_)) {
            
            Integer_Type c_nitems = C.size();   
            for(uint32_t i = 0; i < c_nitems; i++)
            {
                if(not C[i]) 
                    c_sum_local++;
            }
            if(c_sum_local == c_nitems)
                c_sum_local = 1;
            else
                c_sum_local = 0;
        }
        else {
            //if(stationary) {
                uint32_t yi = accu_segment_row;
                auto& iv_data = (*IV)[yi];
                auto& regular_rows = (*rowgrp_regular_rows);
                Integer_Type r_nitems = regular_rows.size();
                for(Integer_Type i: regular_rows) {
                    if(not C[i]) 
                        c_sum_local++;
                }
                if(c_sum_local == r_nitems)
                    c_sum_local = 1;
                else
                    c_sum_local = 0;
            //}
           // else {
                /*
                uint32_t yi = accu_segment_row;
                auto& iv_data = (*IV)[yi];
                auto& regular_rows = (*rowgrp_regular_rows);
                Integer_Type r_nitems = regular_rows.size();
                for(Integer_Type i: regular_rows) {
                    if(not C[i]) 
                        c_sum_local++;
                }
                if(c_sum_local == r_nitems)
                    c_sum_local = 1;
                else
                    c_sum_local = 0;
                */
                
                
                /*
                auto& IR = (*rowgrp_nnz_rows);
                Integer_Type IR_nitems = IR.size();
                Integer_Type i = 0;
                for(Integer_Type i: IR) {
                    if(not C[i]) 
                        c_sum_local++;
                }
                
                if(c_sum_local == IR_nitems)
                    c_sum_local = 1;
                else
                    c_sum_local = 0;
                */
               
            //}
            
            
            /*
            uint32_t yi = accu_segment_row;
            auto& iv_data = (*IV)[yi];
            auto& regular_rows = (*rowgrp_regular_rows);
            Integer_Type r_nitems = regular_rows.size();
            for(Integer_Type i: regular_rows) {
                if(not C[i]) 
                    c_sum_local++;
            }
            if(c_sum_local == r_nitems)
                c_sum_local = 1;
            else
                c_sum_local = 0;
        
            if(not stationary and c_sum_local) {
                c_sum_local = 0;
                auto& source_rows = (*rowgrp_source_rows);
                Integer_Type s_nitems = source_rows.size();
                for(Integer_Type i: source_rows) {
                    if(not C[i]) 
                        c_sum_local++;
                }
                if(c_sum_local == s_nitems)
                    c_sum_local = 1;
                else
                    c_sum_local = 0;
            }
            */
                //MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
                //if(c_sum_gloabl == (uint64_t) Env::nranks)
                //    converged = true;
        }
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
        if(c_sum_gloabl == (uint64_t) Env::nranks)
            converged = true;
    //}
    //else {
        /*
            uint64_t c_sum_local = 0, c_sum_gloabl = 0;
            Integer_Type c_nitems = C.size();   
            for(uint32_t i = 0; i < c_nitems; i++)
            {
                if(not C[i]) 
                    c_sum_local++;
            }
            MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
            if(c_sum_gloabl == (tile_height * Env::nranks))
                converged = true;
        */
        
        
        /*
        Integer_Type c_sum_local = 0, c_sum_gloabl = 0;
        uint32_t yi = accu_segment_row;
        auto& iv_data = (*IV)[yi];
        auto& regular_rows = (*rowgrp_regular_rows);
        Integer_Type r_nitems = regular_rows.size();

        
        for(Integer_Type i: regular_rows) {
            Vertex_State &state = V[i];
            if(not C[i]) 
                c_sum_local++;
        }
        //if(c_sum_local == r_nitems) {
            auto& source_rows = (*rowgrp_source_rows);
            Integer_Type s_nitems = source_rows.size();
            for(Integer_Type i: source_rows) {
                Vertex_State &state = V[i];
                if(not C[i]) 
                    c_sum_local++;
            }
            printf("%d %d %d %d %d\n", Env::rank, c_sum_local, r_nitems, s_nitems, (r_nitems + s_nitems));
            if(c_sum_local == (r_nitems + s_nitems))
                c_sum_local = 1;
            else
                c_sum_local = 0;
        //} 
        //else 
            //c_sum_local = 0;
        
        MPI_Allreduce(&c_sum_local, &c_sum_gloabl, 1, TYPE_INT, MPI_SUM, Env::MPI_WORLD);
        if(c_sum_gloabl == (Integer_Type) Env::nranks)
            converged = true;
        */
    //}        
    return(converged);   
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::checksum()
{
    uint64_t v_sum_local = 0, v_sum_global = 0;
    //Fractional_Type v_sum_local = 0, v_sum_global = 0;
    
    Integer_Type v_nitems = V.size();
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V[i];
        if((state.get_state() != infinity()) and (get_vid(i) < nrows))    
                v_sum_local += state.get_state();
            
    }
    MPI_Allreduce(&v_sum_local, &v_sum_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    //printf("v_sum_local=%lu, %d\n", v_sum_local, Env::rank);
    //MPI_Allreduce(&v_sum_local, &v_sum_global, 1, TYPE_DOUBLE, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
    {
        std::cout << "Iterations: " << iteration << std::endl;
        std::cout << std::fixed << "Value checksum: " << v_sum_global << std::endl;
    }

    uint64_t v_sum_local_ = 0, v_sum_global_ = 0;
    for(uint32_t i = 0; i < v_nitems; i++)
    {
        Vertex_State &state = V[i];
        if((state.get_state() != infinity()) and (get_vid(i) < nrows)) 
            v_sum_local_++;
    }

    MPI_Allreduce(&v_sum_local_, &v_sum_global_, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    if(Env::is_master)
        std::cout << std::fixed << "Reachable vertices: " << v_sum_global_ << std::endl;
    Env::barrier();
}


template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::display(Integer_Type count)
{
    Integer_Type v_nitems = V.size();
    count = (v_nitems < count) ? v_nitems : count;
    Env::barrier();
    
    Triple<Weight, double> stats_pair;
    if(!Env::rank)
    {
        
        #ifdef TIMING
        double sum = 0.0, mean = 0.0, std_dev = 0.0;
        std::cout << "Init           time: " << init_time[0] * 1e3 << " ms" << std::endl;
        stats(scatter_gather_time, sum, mean, std_dev);
        std::cout << "Scatter_gather time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        stats(combine_time, sum, mean, std_dev);
        std::cout << "Combine        time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        stats(apply_time, sum, mean, std_dev);
        std::cout << "Apply          time (sum: avg +/- std_dev): " << sum * 1e3 << ": " << mean * 1e3  << " +/- " << std_dev * 1e3 << " ms" << std::endl;
        std::cout << "Execute        time: " << execute_time[0] * 1e3 << " ms" << std::endl;
        
        std::cout << "TIMING " << init_time[0] * 1e3;
        stats(scatter_gather_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(combine_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        stats(apply_time, sum, mean, std_dev);
        std::cout << " " << sum * 1e3 << " " << mean * 1e3 << " " << std_dev * 1e3;
        std::cout << " " << execute_time[0] * 1e3 << std::endl;
        
        /*
        stats_pair = stats(scatter_gather_time);
        std::cout << "Scatter_gather time  (sum: avg +/- std_dev): " <<  stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_time);
        std::cout << "Combine all     time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_comp_time);
        std::cout << "Combine compute time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(combine_comm_time);
        std::cout << "Combine communi time (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        stats_pair = stats(apply_time);
        std::cout << "Apply time           (sum: avg +/- std_dev): " << stats_pair.row * 1e3  << " ms +/- " << stats_pair.col * 1e3 << " ms" << std::endl;
        */
        #endif
        
        
        Triple<Weight, Integer_Type> pair, pair1;
        for(uint32_t i = 0; i < count; i++)
        {
            pair.row = i;
            pair.col = 0;
            pair1 = A->base(pair, owned_segment, owned_segment);
            Vertex_State &state = V[i];
            //std::cout << std::fixed <<  "vertex[" << A->hasher->unhash(pair1.row) << "]:" << state.print_state() << std::endl;
            std::cout << std::fixed <<  "vertex[" << pair1.row << "]:" << state.print_state() << std::endl;
            //std::cout << std::fixed <<  "vertex[" << pair.row << "]:" << state.print_state() << std::endl;
        }
    }
    Env::barrier();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
void Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::stats(std::vector<double> &vec, double &sum, double &mean, double &std_dev)
{
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type, typename Vertex_State>
struct Triple<Weight, double> Vertex_Program<Weight, Integer_Type, Fractional_Type, Vertex_State>::stats(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
    return{mean, std_dev};
}
#endif
