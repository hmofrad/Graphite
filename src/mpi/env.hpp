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
    static int ncores;     // Number of cores
    static int nsockets;     // Number of sockets
    static int nthreads_per_socket;
    static int nsegments;    // Number of segments = nranks * nthreads
    static std::vector<int> core_ids;
    static std::vector<int> core_ids_unique;
    static std::vector<MPI_Comm> rowgrps_comms;
    static std::vector<MPI_Comm> colgrps_comms; 
    
    static int nmachines; // Number of allocated machines
    static int machine_ncores;
    static int machine_nsocks;
    static std::vector<std::string> machines; // Number of machines
    static std::vector<int> machines_nranks; // Number of ranks per machine
    static std::vector<std::vector<int>> machines_ranks;
    static std::vector<std::vector<int>> machines_cores;
    static std::vector<std::vector<int>> machines_socks;
    static std::vector<std::unordered_set<int>> machines_cores_uniq;
    static std::vector<int> machines_ncores; // Number of cores per machine
    static std::vector<int> machines_nsockets; // Number of sockets available per machine
    
    
    

    private:
        static void affinity(); // Affinity    
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
int Env::ncores = 1;
int Env::nsockets = 1;
int Env::nthreads_per_socket = 1;
int Env::nsegments = 0;
std::vector<int> Env::core_ids;
std::vector<int> Env::core_ids_unique;
std::vector<MPI_Comm> Env::rowgrps_comms; 
std::vector<MPI_Comm> Env::colgrps_comms; 

int Env::nmachines;
int Env::machine_ncores;
int Env::machine_nsocks;
std::vector<std::string> Env::machines;
std::vector<int> Env::machines_nranks;
std::vector<std::vector<int>> Env::machines_ranks;
std::vector<std::vector<int>> Env::machines_cores;
std::vector<std::vector<int>> Env::machines_socks;
std::vector<std::unordered_set<int>> Env::machines_cores_uniq;
std::vector<int> Env::machines_ncores;
std::vector<int> Env::machines_nsockets;


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
    
    // Affinity 
    affinity();
}

void Env::init_threads() {
    int cpu_name_len;
    MPI_Get_processor_name(core_name, &cpu_name_len);
    core_id = sched_getcpu();
    if(core_id == -1)
	{
		fprintf(stderr, "sched_getcpu() returns a negative CPU number\n");
		core_id = 0;
	}
    nthreads = omp_get_max_threads();
    if(numa_available() != -1) {
        ncores = numa_num_configured_cpus();
        nsockets = numa_num_configured_nodes();
        nsockets = (nsockets) ? nsockets : 1;
        nthreads_per_socket = ncores / nsockets;
        nthreads_per_socket = (nthreads_per_socket) ? nthreads_per_socket : 1;
        nsegments = nranks * nthreads;
        if(is_master)
            printf("INFO(rank=%d): nsockets = %d, and nthreads per socket= %d\n", rank, nsockets, nthreads_per_socket);
    }
    else {
        ncores = nthreads;
        nsockets = 1;
        nthreads_per_socket = nthreads / nsockets;
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
    std::sort(core_ids_unique.begin(), core_ids_unique.end());
    core_ids_unique.erase(std::unique(core_ids_unique.begin(), core_ids_unique.end()), core_ids_unique.end());
    //if(!Env::rank) {
    //    for(int i: core_ids)
    //        printf("%d ", i);
    //    printf("\n");
    //}
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



void Env::affinity()
{
    //std::vector<int> core_ids_all = std::vector<int>(nranks * nthreads);
    //MPI_Gather(core_ids.data(), nthreads, MPI_INT, core_ids_all.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); 
    
    std::vector<int> core_ids_all = std::vector<int>(nranks * nthreads);
    MPI_Gather(&core_id, 1, MPI_INT, core_ids_all.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); 
    

    int core_name_len = strlen(core_name);
    int max_length = 0;
    MPI_Allreduce(&core_name_len, &max_length, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
    char* str_padded[max_length + 1]; // + 1 for '\0'
    memset(str_padded, '\0', max_length + 1);
    memcpy(str_padded, &core_name, max_length + 1);
  
    int total_length = (max_length + 1) * Env::nranks; 
    std::string total_string(total_length, '\0');
    MPI_Allgather(str_padded, max_length + 1, MPI_CHAR, (void*) total_string.data(), max_length + 1, MPI_CHAR, MPI_COMM_WORLD);
  
    // Tokenizing the string!
    int offset = 0;
    std::vector<std::string> machines_all;
    machines_all.clear();
    for(int i = 0; i < nranks; i++) 
    {
        machines_all.push_back(total_string.substr(offset, max_length + 1));
        offset += max_length + 1;
    }
  
    if(is_master)
    {
        for(int i = 0; i < nranks; i++) {
            printf("rank=%d, core_id=%d, cpu_name=%s\n", i, core_ids_all[i], machines_all[i].c_str());
            
            //for(int j = 0; j < nthreads; j++) {
            //    printf("rank=%d, core_id=%d, cpu_name=%s\n", i, core_ids_all[j+(i*nthreads)], machines_all[i].c_str());
            //}
        }
    }
    
    nmachines = std::set<std::string>(machines_all.begin(), machines_all.end()).size();
    machines = machines_all; 
    std::vector<std::string>::iterator it;
    it = std::unique(machines.begin(), machines.end());
    machines.resize(std::distance(machines.begin(),it));
    machines_nranks.resize(nmachines, 0);
    machines_ranks.resize(nmachines);
    machines_cores.resize(nmachines);
    machines_socks.resize(nmachines);
    std::vector<std::unordered_set<int>> machines_cores_uniq(nmachines);
    
    
    for (it=machines_all.begin(); it!=machines_all.end(); it++)
    {
        int sz = machines.size();
        int idx = distance(machines.begin(), find(machines.begin(), machines.end(), *it));
        assert((idx >= 0) && (idx < sz));
	  
        machines_nranks[idx]++;
        int idx1 = it - machines_all.begin();

        machines_ranks[idx].push_back(idx1);
        assert((core_ids_all[idx1] >= 0) and (core_ids_all[idx1] < ncores));
        machines_cores[idx].push_back(core_ids_all[idx1]);
        machines_socks[idx].push_back(core_ids_all[idx1] > nsockets ? 0 : 1);
        machines_cores_uniq[idx].insert(core_ids_all[idx1]);
    }  
    
    machines_ncores.resize(nmachines, 0);
    machines_nsockets.resize(nmachines, 0);
    std::vector<int> sockets_per_machine(nsockets, 0);
    for(int i = 0; i < nmachines; i++)
    {
        std::unordered_set<int>::iterator it1;
		for(it1 = machines_cores_uniq[i].begin(); it1 != machines_cores_uniq[i].end(); it1++)
		{
			int socket_id = *it1 / nsockets;
			sockets_per_machine[socket_id] = 1;
			//if(!rank)
			//    std::cout << i << " " << *it1 << " " << socket_id << ", ";
		}
		//if(!rank)
		// std::cout << "\n";
		machines_ncores[i] = machines_cores_uniq[i].size();
		machines_nsockets[i] = std::accumulate(sockets_per_machine.begin(), sockets_per_machine.end(), 0);
    }
    
    machine_ncores = machines_cores[0].size();
    machine_nsocks = nsockets;
    
    if(is_master) 
    {
        std::vector<int>::iterator it1;
        std::vector<int>::iterator it2;
        for(int i = 0; i < nmachines; i++)
        {
            std::cout << "Machine " << i << "=[" << machines[i] << "]";
            std::cout << "| machine_nranks=" << machines_nranks[i];
            std::cout << "| machine_ncores=" << machines_ncores[i];
            std::cout << "| machine_nsockets=" << machines_nsockets[i] << "\n";
            std::cout << "Machine " << i << "=[rank,core,socket]: " ;
            int sz = machines_ranks[i].size();
            for(int j= 0; j < sz; j++) 
            {
                std::cout << "[" << machines_ranks[i][j] <<  "," << machines_cores[i][j] << "," << machines_socks[i][j] << "]";
            }
            std::cout << "\nMachine " << i << "=unique_core(s)[core]:";
            std::unordered_set<int>::iterator iter;
            for(iter=machines_cores_uniq[i].begin(); iter!=machines_cores_uniq[i].end();++iter)
            {
                std::cout << "[" << *iter << "]";
            }
            std::cout << "\n";
        }
    }
    
    
    Env::barrier();
    Env::exit(0);
    
    /*

    nmachines = std::set<std::string>(machines_all.begin(), machines_all.end()).size();
    machines = machines_all; 
    std::vector<std::string>::iterator it;
    it = std::unique(machines.begin(), machines.end());
    machines.resize(std::distance(machines.begin(),it));
    machines_nranks.resize(nmachines, 0);
    machines_ranks.resize(nmachines);
    machines_cores.resize(nmachines);
    machines_socks.resize(nmachines);
    std::vector<std::unordered_set<int>> machines_cores_uniq(nmachines);
  
    for (it=machines_all.begin(); it!=machines_all.end(); it++)
    {
        int sz = machines.size();
        int idx = distance(machines.begin(), find(machines.begin(), machines.end(), *it));
        assert((idx >= 0) && (idx < sz));
	  
        machines_nranks[idx]++;
        int idx1 = it - machines_all.begin();

        machines_ranks[idx].push_back(idx1);
        assert((core_ids[idx1] >= 0) and (core_ids[idx1] < NUM_CORES_PER_MACHINE));
        machines_cores[idx].push_back(core_ids[idx1]);
        machines_socks[idx].push_back(core_ids[idx1] > NUM_CORES_PER_SOCKET ? 0 : 1);
        machines_cores_uniq[idx].insert(core_ids[idx1]);
    }  
    
    machines_ncores.resize(nmachines, 0);
    machines_nsockets.resize(nmachines, 0);
    std::vector<int> sockets_per_machine(NUM_SOCKETS, 0);
    for(int i = 0; i < nmachines; i++)
    {
        std::unordered_set<int>::iterator it1;
		for(it1=machines_cores_uniq[i].begin(); it1!=machines_cores_uniq[i].end();it1++)
		{
			int socket_id = *it1 / NUM_CORES_PER_SOCKET;
			sockets_per_machine[socket_id] = 1;
			//if(!rank)
			//    std::cout << i << " " << *it1 << " " << socket_id << ", ";
		}
		//if(!rank)
		// std::cout << "\n";
		machines_ncores[i] = machines_cores_uniq[i].size();
		machines_nsockets[i] = std::accumulate(sockets_per_machine.begin(), sockets_per_machine.end(), 0);
    }
    
    machine_ncores = machines_cores[0].size();
    machine_nsocks = NUM_SOCKETS;
    
    Env::barrier();
    if(is_master) 
    {
        std::vector<int>::iterator it1;
        std::vector<int>::iterator it2;
        for(int i = 0; i < nmachines; i++)
        {
            std::cout << "Machine " << i << "=[" << machines[i] << "]";
            std::cout << "| machine_nranks=" << machines_nranks[i];
            std::cout << "| machine_ncores=" << machines_ncores[i];
            std::cout << "| machine_nsockets=" << machines_nsockets[i] << "\n";
            std::cout << "Machine " << i << "=[rank,core,socket]: " ;
            int sz = machines_ranks[i].size();
            for(int j= 0; j < sz; j++) 
            {
                std::cout << "[" << machines_ranks[i][j] <<  "," << machines_cores[i][j] << "," << machines_socks[i][j] << "]";
            }
            std::cout << "\nMachine " << i << "=unique_core(s)[core]:";
            std::unordered_set<int>::iterator iter;
            for(iter=machines_cores_uniq[i].begin(); iter!=machines_cores_uniq[i].end();++iter)
            {
                std::cout << "[" << *iter << "]";
            }
            std::cout << "\n";
        }
    }
    
    */
    //Env::barrier();
    //Env::exit(0);
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
    rowgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        grps_init(rowgrps_ranks, rowgrps_nranks, rank_rg, nranks_rg, rowgrps_group_, rowgrps_group, rowgrps_comms[i]);
    }
}

void Env::colgrps_init(std::vector<int32_t>& colgrps_ranks, int32_t colgrps_nranks) {
    grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_group_, colgrps_group, colgrps_comm);   
    colgrps_comms.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        grps_init(colgrps_ranks, colgrps_nranks, rank_cg, nranks_cg, colgrps_group_, colgrps_group, colgrps_comms[i]);  
    }
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
