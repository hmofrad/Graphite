/*
 * 2d.cpp: Tiling unit tests
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
//Compile and run using: g++ -o 2d 2d.cpp && ./2d 4 2
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
};

int p;
int t; 
int nrowgrps;
int ncolgrps;
int rowgrp_nranks;
int colgrp_nranks;
int rank_nrowgrps;
int rank_ncolgrps;
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
                printf("%2d ", tile.thread);
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
            for(auto u: uniques)
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
  //  printf("GCD= %d %d\n", std::gcd(rowgrp_nranks, colgrp_nranks), rank_nrowgrps);
    //int gcd = std::gcd(rank_nrowgrps, rank_ncolgrps);
    int gcd = std::gcd(rowgrp_nranks, colgrp_nranks);
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
        for(int i = 0; i < nrowgrps; i++) {
            
           // printf("%d %d\n", (i/(nrowgrps/(gcd * t))), 
           // (i / (nrowgrps/(gcd * t))) * (rank_nrowgrps/t));
            
            for(int j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j]; 
                tile.rank = (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd*t))) * (rank_nrowgrps/t))) % p;
                //tile.rank += ((i / (nrowgrps/gcd)) * rank_nrowgrps);
                //tile.rank %= p;
                tile.thread = (i / colgrp_nranks) % t;
            }
            
        }
        
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
    print("thread"); 
   print("rank");
//printf("XXXXXXXX\n");    
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
    tiles.resize(p*t);
    for(int i = 0; i < p*t; i++)
        tiles[i].resize(p*t);
    
   // printf("p=%d, nrowgrps      x ncolgrps      = %d x %d\n", p, nrowgrps, ncolgrps);
   // printf("p=%d, rank_nrowgrps x rank_ncolgrps = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
   // printf("p=%d, rowgrp_nranks x colgrp_nranks = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);
    
    bool ret = false;
    ret = TWOD_TStaggered_NEW();
    
    return(ret);
}






