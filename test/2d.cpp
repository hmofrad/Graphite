/*
 * 2d.cpp: Tiling unit tests
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
//Compile and run using: g++ -std=c++17 -o 2d 2d.cpp && ./2d 4 2
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
    /*
    for(int i = 0; i < nrowgrps; i++) {
        printf("%d ", tiles[i][i].rank);
    }
    printf("\n");
    */
    
    bool ret = true;
    std::vector<int> uniques(p);
    for(int i = 0; i < nrowgrps; i++) {
        int r = tiles[i][i].rank;
        uniques[r]++;
    }
    
    for(int i = 0; i < p; i++) {
        if(uniques[i] != t) {
            printf("Processes: \n");
            for(auto u: uniques)
                printf("%d ", u);
            printf("\n");
            ret = false;
            break;
        }
    }
    
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
    
    /*
    std::vector<int> uniques;
    for(int i = 0; i < nrowgrps; i++) {
        int r = tiles[i][i].rank;
        if(std::find(uniques.begin(), uniques.end(), r) == uniques.end()) {
            uniques.push_back(r);
        }
    }
    if(uniques.size() != p) {
        printf("Diagonal ranks are not unique: ");
        std::sort(uniques.begin(), uniques.end());
        for(auto u: uniques)
            printf("%d ", u);
        printf("\n");
        ret = false;
    }
    */
    return(ret);
}
/* Simplidied 2D-thread-based-Staggered 
   See Matrix::init_matrix() method for a
   complete implementation.
*/
bool TWOD_TStaggered() {
    //printf("2D-TStaggered\n");
    
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
            tile.thread = (i / colgrp_nranks) % t;
        }
    }
 //   print("rank"); 
    
    std::vector<int32_t> counts(p);
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < nrowgrps; j++) {
           if(counts[tiles[j][i].rank] < t) {
               counts[tiles[j][i].rank]++;
               if(i != j) {
                   std::swap(tiles[i], tiles[j]);
		   printf("%d <--> %d\n", i, j); 
               }
               break;
           }
        }
    }
    print("rank"); 
    //print("thread"); 
    bool ret = check_diagonals();
    return(ret);
    
}

bool TWOD_TStaggered_NEW() {
    //printf("2D-TStaggered\n");
    printf("GCD= %d %d\n", std::gcd(rowgrp_nranks, colgrp_nranks), rank_nrowgrps);
    //int gcd = std::gcd(rank_nrowgrps, rank_ncolgrps);
    int gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    int gcd_t = std::gcd(rowgrp_nthreads, colgrp_nthreads);
    /*
    if(gcd == 1) {
        for(int i = 0; i < nrowgrps; i++) {
            for(int j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j]; 
                tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
                
            }
        }
    }
    else {
      */  
      //printf("%d %d %d\n", p, t, (p*t) %p);
        for(int i = 0; i < nrowgrps; i++) {
            
           // printf("%d %d\n", (i/(nrowgrps/(gcd * t))), 
           // (i / (nrowgrps/(gcd * t))) * (rank_nrowgrps/t));
            
            for(int j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j]; 
                tile.rank = (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd_r*t))) * (rank_nrowgrps/t))) % p;
                //tile.rank += ((i / (nrowgrps/gcd)) * rank_nrowgrps);
                //tile.rank %= p;
                tile.thread = (i / colgrp_nranks) % t;
                tile.thread_rank = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % (p*t);
                tile.rank_thread = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % p;
            }
        }
        /*
        for(int i = 0; i < nrowgrps; i++) {
            for(int j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j]; 
                tile.rank_thread = tile.thread_rank % p;
            }
        }
        */
        
    //}
        
        
    /*
    for(int i = 0; i < nrowgrps; i++) {
          //  printf("%d %d\n", (i % ncolgrps/rank_ncolgrps), 
            //(i % ncolgrps/rank_ncolgrps) * rank_nrowgrps);
            for(int j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j]; 
                tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
                tile.rank += (i % ncolgrps/rank_ncolgrps) * rank_nrowgrps;
                tile.rank %= p;
                //tile.thread = (i / colgrp_nranks) % t;
                
            }
        }
    }
    */
    
   // print("rank"); 
    /*
    std::vector<int32_t> counts(p);
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < nrowgrps; j++) {
           if(counts[tiles[j][i].rank] < t) {
               counts[tiles[j][i].rank]++;
               if(i != j) {
                   std::swap(tiles[i], tiles[j]);
		   printf("%d <--> %d\n", i, j); 
               }
               break;
           }
        }
    }
    print("rank"); 
    */
    printf("rank\n");
    print("rank");
    printf("thread\n");
    print("thread"); 
    printf("thread_rank\n");
    print("thread_rank");
    printf("rank_thread\n");
    print("rank_thread");
    
    bool ret = check_diagonals();
    return(ret);
    
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <numProcesses> <numThreads>\n", argv[0]);
        exit(0);
    }

    p = atoi(argv[1]);
    t = atoi(argv[2]); 
    
    nrowgrps = p * t;
    ncolgrps = p * t;
    rowgrp_nranks;
    colgrp_nranks;
    integer_factorize(p, rowgrp_nranks, colgrp_nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    integer_factorize(p*t, rowgrp_nthreads, colgrp_nthreads);
    thread_nrowgrps = nrowgrps / colgrp_nthreads;
    thread_ncolgrps = ncolgrps / rowgrp_nthreads;
    tiles.resize(p*t);
    for(int i = 0; i < p*t; i++)
        tiles[i].resize(p*t);
    
    printf("p=%d, nrowgrps        x ncolgrps          = %d x %d\n", p, nrowgrps, ncolgrps);
    printf("p=%d, rank_nrowgrps   x rank_ncolgrps     = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
    printf("p=%d, rowgrp_nranks   x colgrp_nranks     = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);
    printf("p=%d, thread_nrowgrps x thread_ncolgrps   = %d x %d\n", p, thread_nrowgrps, thread_ncolgrps);
    printf("p=%d, rowgrp_nthreads x colgrp_nthreads   = %d x %d\n", p, rowgrp_nthreads, colgrp_nthreads);
    
    bool ret = false;
    ret = TWOD_TStaggered_NEW();
    
    return(ret);
}






