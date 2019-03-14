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
#include <sys/time.h>

#include <mpi.h>
#include <omp.h>
#include <numa.h>
#include <thread>
//#include </ihome/rmelhem/moh18/numactl/libnuma/usr/local/include/numa.h>

struct topology {
    int nmachines;
    int machine_nranks;
    std::vector<struct machine> machines;
};

struct machine {
    std::vector<int> ranks;
    std::vector<int> sockets;
    int nsockets;
    int socket_nranks;
    std::string name;
    std::vector<std::vector<int>> socket_ranks;
};

class Env {
    public:
        Env();
        
        static MPI_Comm MPI_WORLD;
        static int rank;
        static int nranks;
        static int socket_id;
        static bool is_master;
        static void init();
        static void init_threads();
        static void barrier();
        static void finalize();
        static void exit(int code);
        static bool affinity(); // Affinity    
        
        static bool initialized;//, already_initialized;
        static MPI_Group rowgrps_group_, rowgrps_group;
        static std::vector<MPI_Group> rowgrps_groups_, rowgrps_groups;
        static MPI_Comm rowgrps_comm;         
        static int rank_rg;
        static int nranks_rg;
        static MPI_Group colgrps_group_, colgrps_group;
        static std::vector<MPI_Group> colgrps_groups_, colgrps_groups;
        static MPI_Comm colgrps_comm;         
        static int rank_cg;
        static int nranks_cg;
        static void grps_init(std::vector<int32_t>& grps_ranks, int32_t grps_nranks, 
                              int& grps_rank_, int& grps_nranks_, MPI_Group& grps_group_, MPI_Group& grps_group, MPI_Comm& grps_comm);
        static void rowgrps_init(std::vector<int32_t>& rowgrps_ranks, int32_t rowgrps_nranks);
        static void colgrps_init(std::vector<int32_t>& colgrps_ranks, int32_t colgrps_nranks);               

        static double clock();
        static void   print_time(std::string preamble, double time);
        static void   print_num(std::string preamble, uint32_t num);
        static bool   get_init_status();
        static void   set_init_status();
        static int set_thread_affinity(int thread_id);
        static int socket_of_cpu(int cpu_id);
        static int socket_of_thread(int thread_id);
        
        static void shuffle_ranks();
        
        static char core_name[]; // Core name = hostname of MPI rank
        static int core_id;      // Core id of MPI rank
        static int nthreads;     // Number of threads
        static int ncores;     // Number of cores
        static int nsockets;     // Number of sockets
        static int ncores_per_socket;
        static int nsegments;    // Number of segments = nranks * nthreads
        static std::vector<int> core_ids;
        static std::vector<int> core_ids_unique;
        static std::vector<MPI_Comm> rowgrps_comms;
        static std::vector<MPI_Comm> colgrps_comms; 
        
        static std::vector<int> ranks; 
        static struct topology network;

};

MPI_Comm Env::MPI_WORLD;
int  Env::rank = -1;
int  Env::nranks = -1;
bool Env::is_master = false;

bool Env::initialized = false;
//bool Env::already_initialized = false;

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
int Env::ncores = 1;
int Env::nsockets = 1;
int Env::ncores_per_socket = 1;
int Env::nsegments = 0;
std::vector<int> Env::core_ids;
std::vector<int> Env::core_ids_unique;
int Env::socket_id = 0;

std::vector<MPI_Group> Env::rowgrps_groups_;
std::vector<MPI_Group> Env::rowgrps_groups;
std::vector<MPI_Comm> Env::rowgrps_comms; 
std::vector<MPI_Group> Env::colgrps_groups_;
std::vector<MPI_Group> Env::colgrps_groups;
std::vector<MPI_Comm> Env::colgrps_comms; 

std::vector<int> Env::ranks;
struct topology Env::network;

void Env::init() {
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;
    MPI_Init_thread(nullptr, nullptr, required, &provided);
    assert((provided >= MPI_THREAD_SINGLE) && (provided <= MPI_THREAD_MULTIPLE));
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks >= 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);
    is_master = rank == 0;
    
    init_threads();
    
    if(required != provided) {
        printf("WARN(rank=%d): Failure to enable MPI_THREAD_MULTIPLE(%d) for multithreading\n", rank, provided); 
        nthreads = 1;
    }
    
    MPI_WORLD = MPI_COMM_WORLD;
    
    printf("INFO(rank=%d): Hostname=%s, core_id=%d, nthreads=%d\n", rank, core_name, core_id, nthreads);
    MPI_Barrier(MPI_COMM_WORLD);   
}

void Env::init_threads() {
    int cpu_name_len;
    MPI_Get_processor_name(core_name, &cpu_name_len);
    core_id = sched_getcpu();
    
    if(core_id == -1) {
		fprintf(stderr, "sched_getcpu() returns a negative CPU number\n");
		core_id = 0;
	}
    
    if(numa_available() != -1) {
        omp_set_dynamic(0);
        nthreads = omp_get_max_threads();
        ncores = numa_num_configured_cpus();
        nsockets = numa_num_configured_nodes();
        nsockets = (nsockets) ? nsockets : 1;
        ncores_per_socket = ncores / nsockets;
        ncores_per_socket = (ncores_per_socket) ? ncores_per_socket : 1;
        nsegments = nranks * nthreads;
        if(is_master)
            printf("INFO(rank=%d): nsockets = %d, and nthreads per socket= %d\n", rank, nsockets, ncores_per_socket);
    }
    else {
        nthreads = 1;
        ncores = 1;
        nsockets = 1;
        omp_set_dynamic(0);
        omp_set_num_threads(nthreads);
        ncores_per_socket = nthreads / nsockets;
        nsegments = nranks * nthreads;
        printf("WARN(rank=%d): Failure to enable NUMA-aware memory allocation\n", rank);
    }
    
    core_ids.resize(Env::nthreads);
    core_ids_unique.resize(Env::nthreads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        core_ids[tid] = sched_getcpu();
        if(core_ids[tid] == -1)
        {
            fprintf(stderr, "sched_getcpu() returns a negative CPU number");
            core_ids[tid] = 0;
        }
        core_ids_unique[tid] = core_ids[tid];
    }
    std::sort(core_ids.begin(), core_ids.end());
    std::sort(core_ids_unique.begin(), core_ids_unique.end());
    core_ids_unique.erase(std::unique(core_ids_unique.begin(), core_ids_unique.end()), core_ids_unique.end());
    socket_id = socket_of_cpu(core_id);
}

int Env::set_thread_affinity(int thread_id) {
    int num_unique_cores = core_ids_unique.size();
    int cid = core_ids_unique[thread_id % num_unique_cores];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cid, &cpuset);
    pthread_t current_thread = pthread_self();    
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

int Env::socket_of_cpu(int cpu_id) {
    return(cpu_id / ncores_per_socket);
}

int Env::socket_of_thread(int thread_id) {
    int num_unique_cores = core_ids_unique.size();
    int cid = core_ids_unique[thread_id % num_unique_cores];
    return(socket_of_cpu(cid));
}

bool Env::affinity() {   
    bool enabled = true;
    shuffle_ranks();
    // Get information about cores a rank owns
    std::vector<int> core_ids_all = std::vector<int>(nranks * nthreads);
    MPI_Allgather(core_ids.data(), nthreads, MPI_INT, core_ids_all.data(), nthreads, MPI_INT, MPI_WORLD); 
    // Get machine names in machines_all
    int core_name_len = strlen(core_name);
    int max_length = 0;
    MPI_Allreduce(&core_name_len, &max_length, 1, MPI_INT, MPI_MAX, MPI_WORLD);
    char* str_padded[max_length + 1]; // + 1 for '\0'
    memset(str_padded, '\0', max_length + 1);
    memcpy(str_padded, &core_name, max_length + 1);
    int total_length = (max_length + 1) * Env::nranks; 
    std::string total_string(total_length, '\0');
    MPI_Allgather(str_padded, max_length + 1, MPI_CHAR, (void*) total_string.data(), max_length + 1, MPI_CHAR, MPI_WORLD);
    // Tokenizing the string!
    int offset = 0;
    std::vector<std::string> machines_all;
    //machines_all.clear();
    for(int i = 0; i < nranks; i++) {
        machines_all.push_back(total_string.substr(offset, max_length + 1));
        offset += max_length + 1;
    }
    /*
    Env::barrier();
    if(!Env::rank) {
        for(int i = 0; i < nranks; i++) {
            printf("[%d, %s]: ", i, machines_all[i].c_str());
            for(int j = 0; j < nthreads; j++) {
                printf("%d ", core_ids_all[(i*nthreads) + j]);
            }
            printf("\n");
        }
    }
    Env::barrier();
    */
    // Find unique machines
    std::vector<std::string> machines = machines_all; 
    sort(machines.begin(), machines.end());
    machines.erase(unique(machines.begin(), machines.end()), machines.end()); 
    int nmachines = machines.size();
    int machine_nranks = nranks / nmachines; 
    network.machine_nranks = machine_nranks;
    int socket_nranks = machine_nranks / nsockets;
    network.nmachines = nmachines;
    network.machines.resize(nmachines);
    for(int i = 0; i < nmachines; i++) {
        auto& machine = network.machines[i];
        machine.name = machines[i];
        machine.socket_ranks.resize(nsockets);
        machine.socket_nranks = socket_nranks;
    }
   
    std::vector<std::string>::iterator it;
    int i = 0, j = 0;
    int cid = 0, sid = 0;
    for (it = machines_all.begin(); it != machines_all.end(); it++) {
        int idx = distance(machines.begin(), find(machines.begin(), machines.end(), *it));
        int idx1 = it - machines_all.begin();
        network.machines[idx].ranks.push_back(idx1);
        for(j = 0; j < nthreads; j++) {
            cid = core_ids_all[i];
            sid = socket_of_cpu(cid);
            network.machines[idx].socket_ranks[sid].push_back(idx1);
            i++;
        }
        network.machines[idx].sockets.push_back(sid); 
    }  
    
    i = 0;
    ranks.resize(nranks);
    for(auto& machine: network.machines) {
        for(std::vector<int>& socket_ranks_: machine.socket_ranks) {
            std::vector<int> this_socket_ranks = socket_ranks_;
            this_socket_ranks.erase(std::unique(this_socket_ranks.begin(), this_socket_ranks.end()), this_socket_ranks.end());
            for(int rank_: this_socket_ranks) {
                ranks[i] = rank_;
                i++;
            }
        }
    }   

    for(int i = 0; i < nmachines; i++) {
        auto& machine = network.machines[i];
        //assert(network.machine_nranks == (int) machine.ranks.size());
        if(network.machine_nranks != (int) machine.ranks.size()) {
            enabled = false;
            break;
        }
        for(int j = 0; j < nsockets; j++) {
            int uniq = std::set<int>(machine.socket_ranks[j].begin(), machine.socket_ranks[j].end()).size();
            //assert(machine.socket_nranks == uniq);
            if(machine.socket_nranks != uniq) {
                enabled = false;
                break;
            }
        }
        if(not enabled)
            break;
    }  
    if(not enabled) {
        if(is_master)
            printf("WARN(rank=%d): Failure to enable 2D NUMA tiling. Falling back to 2D machine level tiling\n", rank);
    }
    /*
    Env::barrier();
    if(!Env::rank) {
        for(auto& machine: network.machines) {
            printf("%s\n", machine.name.c_str());
            printf("   Ranks  : ");
            for(int rank_: machine.ranks)
                printf("%d ", rank_);
            printf("\n");
            printf("   Sockets: ");
            for(int socket_: machine.sockets)
                printf("%d ", socket_);
            printf("\n");
            for(std::vector<int>& socket_ranks_: machine.socket_ranks) {
                printf("   Sockets ranks: ");
                for(int rank_: socket_ranks_)
                    printf("%d ", rank_);
                printf("\n");
            }
            
        }
        printf("NUMA ranks order: ");
        for(int i = 0; i < nranks; i++) {
            printf("%d ", ranks[i]);
        }
        printf("\n");
    }
    Env::barrier();
    */
    return(enabled);
}



bool Env::get_init_status() {
    return(initialized);
}

void Env::set_init_status() {
    if(not initialized)
        initialized = true;
}


void Env::grps_init(std::vector<int32_t>& grps_ranks, int grps_nranks, int& grps_rank_, int& grps_nranks_,
                    MPI_Group& grps_group_, MPI_Group& grps_group, MPI_Comm& grps_comm) {
    
    MPI_Comm_group(MPI_WORLD, &grps_group_);
    MPI_Group_incl(grps_group_, grps_nranks, grps_ranks.data(), &grps_group);
    MPI_Comm_create(MPI_WORLD, grps_group, &grps_comm);
    
    if (MPI_COMM_NULL != grps_comm) 
    {
        MPI_Comm_rank(grps_comm, &grps_rank_);
        MPI_Comm_size(grps_comm, &grps_nranks_);
    }
}

void Env::rowgrps_init(std::vector<int32_t>& rowgrps_ranks, int32_t rowgrps_nranks) {
    grps_init(rowgrps_ranks, rowgrps_nranks, rank_rg, nranks_rg, rowgrps_group_, rowgrps_group, rowgrps_comm);
    
    rowgrps_groups_.resize(Env::nthreads);
    rowgrps_groups.resize(Env::nthreads);
    rowgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        grps_init(rowgrps_ranks, rowgrps_nranks, rank_rg, nranks_rg, rowgrps_groups_[i], rowgrps_groups[i], rowgrps_comms[i]);
    }
    
    /*
    rowgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        MPI_Comm_split(MPI_COMM_WORLD, i, rank, &rowgrps_comms[i]);
        MPI_Comm_rank(rowgrps_comms[i], &rank_rg);
        MPI_Comm_size(rowgrps_comms[i], &nranks_rg);        
    }
    */
}

void Env::colgrps_init(std::vector<int32_t>& colgrps_ranks, int32_t colgrps_nranks) {
    grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_group_, colgrps_group, colgrps_comm);   
    
    colgrps_groups_.resize(Env::nthreads);
    colgrps_groups.resize(Env::nthreads);
    colgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_groups_[i], colgrps_groups[i], colgrps_comms[i]);  
    }
    
    /*
    colgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        MPI_Comm_split(MPI_COMM_WORLD, i, rank, &colgrps_comms[i]);
        MPI_Comm_rank(colgrps_comms[i], &rank_cg);
        MPI_Comm_size(colgrps_comms[i], &nranks_cg);        
    }
    */
}

void Env::shuffle_ranks()
{
  std::vector<int> ranks(nranks);
  if (is_master)
  {
    srand(time(NULL));
    std::iota(ranks.begin(), ranks.end(), 0);  // ranks = range(len(ranks))
    std::random_shuffle(ranks.begin() + 1, ranks.end());

    assert(ranks[0] == 0);
  }

  MPI_Bcast(ranks.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Group world_group, reordered_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, nranks, ranks.data(), &reordered_group);
  MPI_Comm_create(MPI_COMM_WORLD, reordered_group, &MPI_WORLD);

  MPI_Comm_rank(MPI_WORLD, &rank);
  MPI_Comm_size(MPI_WORLD, &nranks);
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
    
    MPI_Group_free(&rowgrps_group_);
    MPI_Group_free(&rowgrps_group);
    MPI_Comm_free(&rowgrps_comm);
    
    for(int i = 0; i < Env::nthreads; i++) {
        MPI_Group_free(&rowgrps_groups_[i]);
        MPI_Group_free(&rowgrps_groups[i]);
        MPI_Comm_free(&rowgrps_comms[i]);
    }
    
    MPI_Group_free(&colgrps_group_);
    MPI_Group_free(&colgrps_group);
    MPI_Comm_free(&colgrps_comm);   
    
    for(int i = 0; i < Env::nthreads; i++) {
        MPI_Group_free(&colgrps_groups_[i]);
        MPI_Group_free(&colgrps_groups[i]);
        MPI_Comm_free(&colgrps_comms[i]);
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
