/*
 * 2d.cpp: Tiling unit tests
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
// Compile and run using: g++ -std=c++17 -o 2d 2d.cpp && ./2d 4 2 // 4 processes each with 2 threads
// Unit test using: ./2d_test.sh

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include  <numeric>


struct Tile2D { 
    int rank;
    int thread;
    int thread_rank;
    int rank_thread;
};

int p;
int t; 
int nrowgrps;
int ncolgrps;
int rowgrp_nranks;
int colgrp_nranks;
int rank_nrowgrps;
int rank_ncolgrps;
int rowgrp_nthreads;
int colgrp_nthreads;
int thread_nrowgrps;
int thread_ncolgrps;

std::string tiling_type;

std::vector<std::vector<struct Tile2D>> tiles;

void integer_factorize(int n, int& a, int& b) {
    a = b = sqrt(n);
    while (a * b != n) {
        b++;
        a = n / b;
    }
    
    if((a * b) != n) {
        printf("Assetion failed for [n=%d] == [a=%d, b=%d]\n", n, a, b);
    }
}

void print(std::string field){
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            if(field.compare("rank") == 0) 
                printf("%02d ", tile.rank);
            else if(field.compare("thread") == 0) 
                printf("%02d ", tile.thread);
            else if(field.compare("thread_rank") == 0) 
                printf("%02d ", tile.thread_rank);
            else if(field.compare("rank_thread") == 0) 
                printf("%02d ", tile.rank_thread);
        }
        printf("\n");
    }
    printf("\n");
}

bool check_diagonals() {
    bool ret = true;
       
    if (tiling_type == "TWOD_Staggered") {
        std::vector<int> uniques(p);
        for(int i = 0; i < nrowgrps; i++) {
            int r = tiles[i][i].rank;
            uniques[r]++;
        }
        for(int i = 0; i < p; i++) {
            if(uniques[i] != 1) {
                printf("Processes: \n");
                for(auto u: uniques)
                    printf("%d ", u);
                printf("\n");
                ret = false;
                break;
            }
        }
    }    
    else if (tiling_type == "TWOD_TStaggered_NEW") {
        std::vector<int> uniques_thread_rank(p*t);
        for(int i = 0; i < nrowgrps; i++) {
            int r = tiles[i][i].thread_rank;
            uniques_thread_rank[r]++;
        }
        
        for(int i = 0; i < p*t; i++) {
            if(uniques_thread_rank[i] != 1) {
                printf("Thread_rank: \n");
                for(auto u: uniques_thread_rank)
                    printf("%d ", u);
                printf("\n");
                ret = false;
                break;
            }
        } 


     
        std::vector<int> uniques_rank_thread(p);
        for(int i = 0; i < nrowgrps; i++) {
            int r = tiles[i][i].rank_thread;
            uniques_rank_thread[r]++;
        }
        
        for(int i = 0; i < p; i++) {
            if(uniques_rank_thread[i] != t) {
                printf("Rank_thread: \n");
                for(auto u: uniques_rank_thread)
                    printf("%d ", u);
                printf("\n");
                ret = false;
                break;
            }
        }
        
        std::vector<int> uniques_thread(t);
        for(int i = 0; i < nrowgrps; i++) {
            int r = tiles[i][i].thread;
            uniques_thread[r]++;
        }
        
        for(int i = 0; i < t; i++) {
            if(uniques_thread[i] != p) {
                printf("Threads: \n");
                for(auto u: uniques_thread)
                    printf("%d ", u);
                printf("\n");
                ret = false;
                break;
            }
        }
        

        
    }
    return(ret);
}
/* 2D-process-based-Staggered (LA3) */
bool TWOD_Staggered() {
    nrowgrps = p;
    ncolgrps = p;
    rowgrp_nranks;
    colgrp_nranks;
    integer_factorize(p, rowgrp_nranks, colgrp_nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    
    printf("p=%d, nrowgrps        x ncolgrps          = %d x %d\n", p, nrowgrps, ncolgrps);
    printf("p=%d, rank_nrowgrps   x rank_ncolgrps     = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
    printf("p=%d, rowgrp_nranks   x colgrp_nranks     = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);

    tiles.resize(p);
    for(int i = 0; i < p; i++)
        tiles[i].resize(p);
    
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
        }
    }

    std::vector<int32_t> counts(p);
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < nrowgrps; j++) {
           if(counts[tiles[j][i].rank] < 1) {
               counts[tiles[j][i].rank]++;
               if(i != j) {
                   std::swap(tiles[i], tiles[j]);
               }
               break;
           }
        }
    }
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
    
}
/* 2D-thread-based-Staggered (Graphite)
   See Matrix::init_matrix() method.
*/
bool TWOD_TStaggered_NEW() {
    nrowgrps = p * t;
    ncolgrps = p * t;
    rowgrp_nranks;
    colgrp_nranks;
    integer_factorize(p, rowgrp_nranks, colgrp_nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    int gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    /*
    integer_factorize(p*t, rowgrp_nthreads, colgrp_nthreads);
    thread_nrowgrps = nrowgrps / colgrp_nthreads;
    thread_ncolgrps = ncolgrps / rowgrp_nthreads;
    int gcd_t = std::gcd(rowgrp_nthreads, colgrp_nthreads);
    */
    rowgrp_nthreads = rowgrp_nranks;
    colgrp_nthreads = nrowgrps / rowgrp_nthreads;
    thread_nrowgrps = nrowgrps / colgrp_nthreads;
    thread_ncolgrps = ncolgrps / rowgrp_nthreads;
    int gcd_t = std::gcd(rowgrp_nthreads, colgrp_nthreads);
    
    
    /*
    printf("p=%d, nrowgrps        x ncolgrps          = %d x %d\n", p, nrowgrps, ncolgrps);
    printf("p=%d, rank_nrowgrps   x rank_ncolgrps     = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
    printf("p=%d, rowgrp_nranks   x colgrp_nranks     = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);
    printf("p=%d, thread_nrowgrps x thread_ncolgrps   = %d x %d\n", p, thread_nrowgrps, thread_ncolgrps);
    printf("p=%d, rowgrp_nthreads x colgrp_nthreads   = %d x %d\n", p, rowgrp_nthreads, colgrp_nthreads);
    */
    tiles.resize(p*t);
    for(int i = 0; i < p*t; i++)
        tiles[i].resize(p*t);

    for(int i = 0; i < nrowgrps; i++) {        
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rank = (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd_r*t))) * (rank_nrowgrps/t))) % p;
            tile.thread_rank = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % (p*t);
            //tile.thread_rank = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % (p*t);
            //tile.rank_thread = tile.thread_rank%p;
            //tile.rank_thread = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % p;
            tile.rank_thread = (tile.thread_rank  % p);
            //tile.rank_thread = tile.thread_rank % p;
            //tile.thread = (i / colgrp_nranks) % t;
            tile.thread = tile.thread_rank / p;
            
        }
    }
    
    /*
    printf("rank\n");
    print("rank");
    printf("thread\n");
    print("thread"); 
    printf("thread_rank\n");
    print("thread_rank");
    printf("rank_thread\n");
    print("rank_thread");
    */
    /*
    std::vector<std::vector<int>> assignment;
    assignment.resize(p * t);
    for(int i = 0; i < nrowgrps; i++) {        
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            auto& a = assignment[tile.thread_rank];
              std::vector<int>::iterator it = std::find(a.begin(), a.end(), tile.rank);
              if(it == a.end())
                assignment[tile.thread_rank].push_back(tile.rank);
        }
    }
    
    for(int i = 0; i < p * t; i++) {
        
    }
    
    for(int i = 0; i < p * t; i++) {
        printf("%2d: ", i);
        for(auto aa: assignment[i]) {
            printf("%2d ", aa);
        }
        printf("\n");
    }
    */
    
    bool ret = check_diagonals();
    return(ret);
    
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: %s <numProcesses> <numThreads> <tiling_type>\n", argv[0]);
        exit(0);
    }

    p = atoi(argv[1]);
    t = atoi(argv[2]); 
    

    
    tiling_type = argv[3];
    
    bool ret = false;
    if(!strcmp(argv[3], "TWOD_Staggered")) {
        printf("Process-based 2D-Staggered (LA3)\n");
        ret = TWOD_Staggered();
    }
    else if (!strcmp(argv[3], "TWOD_TStaggered_NEW")) {
        //printf("Thread-based 2D-Staggered (Graphite)\n");
        ret = TWOD_TStaggered_NEW();
    }
    else {
        printf("Incorrect tiling_type = %s\n", argv[3]);
        exit(0);
    }
    
    return(ret);
}






