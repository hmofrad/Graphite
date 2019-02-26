/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef ENV_HPP
#define ENV_HPP
 
#include <mpi.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <sched.h>
#include <unordered_set>
#include <set>
#include <cstring>
#include <numeric>
#include <algorithm>


//#ifdef __linux__
#include <numa.h>
//#endif 
//#include </ihome/rmelhem/moh18/numactl/libnuma/usr/local/include/numa.h>
//#ifdef NUMA_AVAILABLE 

class Env {
    public:
    Env();
    
    static MPI_Comm MPI_WORLD;
    static int rank;
    static int nranks;
    static bool is_master;
    static void init(bool comm_split_ = true);
    static void init_t();
    static void barrier();
    static void finalize();
    static void exit(int code);
    
    static bool comm_split, comm_created; // Splitting the world communicator
    static MPI_Group rowgrps_group_, rowgrps_group;
    static MPI_Comm rowgrps_comm;         
    static int rank_rg;
    static int nranks_rg;
    static MPI_Group colgrps_group_, colgrps_group;
    static MPI_Comm colgrps_comm;         
    static int rank_cg;
    static int nranks_cg;
    static void grps_init(std::vector<int32_t>& grps_ranks, int32_t grps_nranks, 
               int& grps_rank_, int& grps_nranks_,
               MPI_Group& grps_group_, MPI_Group& grps_group, MPI_Comm& grps_comm);
    static void rowgrps_init(std::vector<int32_t>& rowgrps_ranks, int32_t rowgrps_nranks);
    static void colgrps_init(std::vector<int32_t>& colgrps_ranks, int32_t colgrps_nranks);               

    static double clock();
    static void   print_time(std::string preamble, double time);
    static void   print_num(std::string preamble, uint32_t num);
    static void   set_comm_split();
    static bool   get_comm_split();
    
    static char core_name[]; // Core name = hostname
    static int core_id; // Core id
    static int nthreads;
    static int nsockets;
    static int nthreads_per_socket;
    static int nsegments;
};

MPI_Comm Env::MPI_WORLD;
int  Env::rank = -1;
int  Env::nranks = -1;
bool Env::is_master = false;

bool Env::comm_split = true;
bool Env::comm_created = false;

MPI_Group Env::rowgrps_group_;
MPI_Group Env::rowgrps_group;
MPI_Comm Env::rowgrps_comm;
int  Env::rank_rg = -1;
int  Env::nranks_rg = -1;

MPI_Group Env::colgrps_group_;
MPI_Group Env::colgrps_group;   
MPI_Comm Env::colgrps_comm;
int  Env::rank_cg = -1;
int  Env::nranks_cg = -1;

char Env::core_name[MPI_MAX_PROCESSOR_NAME];
int Env::core_id;
int Env::nthreads = 1;
int Env::nsockets = 1;
int Env::nthreads_per_socket = 1;
int Env::nsegments = 0;
 
void Env::init(bool comm_split_) {
    comm_split = comm_split_;
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;
    MPI_Init_thread(nullptr, nullptr, required, &provided);
    assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks >= 0);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);
    
    is_master = rank == 0;
    
    MPI_WORLD = MPI_COMM_WORLD;
    if(required != provided)
    {
        if(is_master)
        {
            printf("Failure to set MPI_THREAD_MULTIPLE by MPI (%d)\n", provided); 
            printf("Multi-threading is disabled with MPI_THREAD_MULTIPLE (%d/%d)\n", omp_get_num_threads(), omp_get_max_threads());
        }
    }
    else
    {
        if(is_master)
            printf("Multi-threading is enabled with MPI_THREAD_MULTIPLE (%d/%d)\n", omp_get_num_threads(), omp_get_max_threads());
    }
    
    init_t();
}

/*
inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }
  */

void Env::init_t() {
    //if(is_master) {
        int cpu_name_len;
        MPI_Get_processor_name(core_name, &cpu_name_len);
        core_id = sched_getcpu();
        printf("Rank=%d is running on %s with core id %d\n", Env::rank, core_name, core_id);
        if(numa_available() != -1) {
        
        //assert( numa_available() != -1 );
        //printf("Initializing threads\n");
            
            nthreads = numa_num_configured_cpus();
            nsockets = numa_num_configured_nodes();
            nsockets = (nsockets) ? nsockets : 1;
            nthreads_per_socket = nthreads / nsockets;
            nthreads_per_socket = (nthreads_per_socket) ? nthreads_per_socket : 1;
            nthreads = omp_get_max_threads();
            
            int nodestring_size = nsockets*2+1;
            char nodestring[nodestring_size];
            memset(nodestring, 0, nodestring_size);
            nodestring[0] = '0';
            for(int i = 1; i < nsockets; i++) {
              nodestring[i*2-1] = ',';
              nodestring[i*2] = '0' + i;
            }
            
            //printf("nodestring=%s %d\n", nodestring, nsockets*2+1);
            struct bitmask * nodemask = numa_parse_nodestring(nodestring);
            numa_set_interleave_mask(nodemask);   
            
            omp_set_dynamic(0);
            //omp_set_num_threads(nthreads);
            if(is_master)
                printf("NUMA-aware OpenMP is enabled with %d threads, %d sockets, and %d threads per socket\n", nthreads, nsockets, nthreads_per_socket);
        }
        else {
            nthreads = omp_get_max_threads();
            nsockets = 1;
            nthreads_per_socket = nthreads / nsockets;
            if(is_master)
                printf("NUMA-aware OpenMP is disabled with %d threads, %d sockets, and %d threads per socket\n", nthreads, nsockets, nthreads_per_socket);
        }
        nsegments = nranks * nthreads;
        /*
        if(is_master) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int cid = sched_getcpu();
                int sid =  cid / nthreads_per_socket;
                int sof =  cid % nthreads_per_socket;
                //int* test = (int*) numa_alloc_onnode(2, sid);
                //test[0] = tid;
                printf("Core_name=%s Rank=%d, tid=%d, core_id=%d, sid=%d sof=%d\n", core_name, Env::rank, tid, cid, sid, sof);
                
            }
        }
        */
        
        
        //for(int i = 0; i < (nsockets*2+1); i++)
        //    printf("%c ", nodestring[i]);
        //printf("\n");

        
        
        
    //}
    
    /*
    //int tid, sid;
    #pragma omp parallel
    {
    //for (int i = 0; i < omp_get_max_threads(); i++) {
        int tid = omp_get_thread_num();
        int sid = get_socket_id(tid);
      //assert(numa_run_on_node(s_i)==0);
      //#ifdef PRINT_DEBUG_MESSAGES
       printf("thread-%d bound to socket-%d\n", tid, sid);
      //#endif
    }
    */

}

bool Env::get_comm_split() {
    return(comm_created);
}

void Env::set_comm_split() {
    if(comm_split)
        comm_created = true;
}

void Env::grps_init(std::vector<int32_t>& grps_ranks, int grps_nranks, int& grps_rank_, int& grps_nranks_,
                    MPI_Group& grps_group_, MPI_Group& grps_group, MPI_Comm& grps_comm) {
    
    MPI_Comm_group(MPI_COMM_WORLD, &grps_group_);
    MPI_Group_incl(grps_group_, grps_nranks, grps_ranks.data(), &grps_group);
    MPI_Comm_create(MPI_COMM_WORLD, grps_group, &grps_comm);
    
    if (MPI_COMM_NULL != grps_comm) 
    {
        MPI_Comm_rank(grps_comm, &grps_rank_);
        MPI_Comm_size(grps_comm, &grps_nranks_);
    }
}

void Env::rowgrps_init(std::vector<int32_t>& rowgrps_ranks, int32_t rowgrps_nranks) {
    grps_init(rowgrps_ranks, rowgrps_nranks, rank_rg, nranks_rg, rowgrps_group_, rowgrps_group, rowgrps_comm);
}

void Env::colgrps_init(std::vector<int32_t>& colgrps_ranks, int32_t colgrps_nranks) {
    grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_group_, colgrps_group, colgrps_comm);   
}

double Env::clock() {
    return(MPI_Wtime());
}

void Env::print_time(std::string preamble, double time) {
    if(is_master)
        printf("%s time: %f seconds\n", preamble.c_str(), time);
}

void Env::print_num(std::string preamble, uint32_t num) {
    if(is_master)
        printf("%s %d\n", preamble.c_str(), num);
}

void Env::finalize() {
    Env::barrier();
    if(Env::comm_split)
    {
        if(rowgrps_group_)
        {
            MPI_Group_free(&rowgrps_group_);
            MPI_Group_free(&rowgrps_group);
            MPI_Comm_free(&rowgrps_comm);
            
            MPI_Group_free(&colgrps_group_);
            MPI_Group_free(&colgrps_group);
            MPI_Comm_free(&colgrps_comm);   
        }
    }
    int ret = MPI_Finalize();
    assert(ret == MPI_SUCCESS);
}

void Env::exit(int code) {    
    Env::finalize();
    std::exit(code);
}

void Env::barrier() {
    MPI_Barrier(MPI_WORLD);
}
#endif
