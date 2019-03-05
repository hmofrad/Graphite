/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef ENV_HPP
#define ENV_HPP
 
#include <cassert>
#include <iostream>
#include <vector>
#include <sched.h>
#include <unordered_set>
#include <set>
#include <cstring>
#include <numeric>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <numa.h>
#include <thread>
//#include </ihome/rmelhem/moh18/numactl/libnuma/usr/local/include/numa.h>

class Env {
    public:
    Env();
    
    static MPI_Comm MPI_WORLD;
    static int rank;
    static int nranks;
    static bool is_master;
    static void init(bool comm_split_ = true);
    static void init_threads();
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
    static int set_thread_affinity(int thread_id);
    
    static char core_name[]; // Core name = hostname of MPI rank
    static int core_id;      // Core id of MPI rank
    static int nthreads;     // Number of threads
    static int nsockets;     // Number of sockets
    static int nthreads_per_socket;
    static int nsegments;    // Number of segments = nranks * nthreads
    static std::vector<int> core_ids;
    
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
std::vector<int> Env::core_ids;
 
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
    
    init_threads();
    
    if(required != provided) {
        printf("WARN(rank=%d): Failure to enable MPI_THREAD_MULTIPLE(%d) for multithreading\n", rank, provided); 
        nthreads = 1;
    }
    printf("INFO(rank=%d): Hostname=%s, core_id=%d, nthreads=%d\n", rank, core_name, core_id, nthreads);
}

void Env::init_threads() {
    int cpu_name_len;
    MPI_Get_processor_name(core_name, &cpu_name_len);
    core_id = sched_getcpu();
    nthreads = omp_get_max_threads();
    if(numa_available() != -1) {
        nsockets = numa_num_configured_nodes();
        nsockets = (nsockets) ? nsockets : 1;
        nthreads_per_socket = numa_num_configured_cpus() / nsockets;
        nthreads_per_socket = (nthreads_per_socket) ? nthreads_per_socket : 1;
        nsegments = nranks * nthreads;
        if(is_master)
            printf("INFO(rank=%d): nsockets = %d, and nthreads per socket= %d\n", rank, nsockets, nthreads_per_socket);
    }
    else {
        nsockets = 1;
        nthreads_per_socket = nthreads / nsockets;
        nsegments = nranks * nthreads;
        printf("WARN(rank=%d): Failure to enable NUMA-aware memory allocation\n", rank);
    }
    
    core_ids.resize(Env::nthreads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        core_ids[tid] = sched_getcpu();
    }
    std::sort(core_ids.begin(), core_ids.end());
    core_ids.erase(std::unique(core_ids.begin(), core_ids.end()), core_ids.end());
    //if(!Env::rank) {
    //    for(int i: core_ids)
    //        printf("%d ", i);
    //    printf("\n");
    //}
}

int Env::set_thread_affinity(int thread_id) {
    int num_unique_cores = core_ids.size();
    int cid = core_ids[thread_id % num_unique_cores];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cid, &cpuset);
    pthread_t current_thread = pthread_self();    
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
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
        printf("INFO(rank=%d): %s time: %f seconds\n", Env::rank, preamble.c_str(), time);
}

void Env::print_num(std::string preamble, uint32_t num) {
    if(is_master)
        printf("INFO(rank=%d): %s %d\n", Env::rank, preamble.c_str(), num);
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
