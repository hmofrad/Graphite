/*
 * 2d.cpp: Tiling unit tests
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
//Compile and run using: g++ -o 2d 2d.cpp && ./2d "2D-Staggered" 4 2
// Unit test using: ./2d_test.sh "2D-Staggered"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>


struct Tile2D { 
    int rg;
    int cg;
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
    //assert(a * b == n);
    if((a * b) != n) {
        printf("Assetion failed for [n=%d] == [a=%d, b=%d]\n", n, a, b);
    }
}

void print(std::string field){
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            if(field.compare("rank") == 0) 
                printf("%d ", tile.rank);
            else if(field.compare("thread") == 0) 
                printf("%d ", tile.thread);
            else if(field.compare("rg") == 0) 
                printf("%d ", tile.rg);
        }
        printf("\n");
    }
    //printf("\n");
}

bool check_diagonals() {
    bool ret = true;
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
    return(ret);
}

bool TWOD_Staggered() {
    //printf("2D-Staggered\n");
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rg = i;
            tile.cg = j;
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
        }
    }
    print("rank"); 
    printf("\n");
    std::vector<int> leader_ranks;
    leader_ranks.resize(nrowgrps, -1);
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < nrowgrps; j++) {
            if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[j][i].rank) != leader_ranks.end())) {
                //if(i != j) {
                //printf("%d %d %d\n", i, j, tiles[j][i].rank);
                if(i != j) {
                    std::swap(tiles[i], tiles[j]);
                }
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
    }
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
}

bool TWOD_TStaggered() {
    //printf("2D-TStaggered\n");
    
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rg = i;
            tile.cg = j;
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
        }
    }
    
    print("rank"); 
    printf("\n");
    std::vector<int32_t> counts(p);
    //std::vector<int> leader_ranks;
    //leader_ranks.resize(nrowgrps, -1);
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < nrowgrps; j++) {
           if(counts[tiles[j][i].rank] < t) {
               counts[tiles[j][i].rank]++;
               if(i != j) {
                 //  printf("Swap(%d %d)\n", i, j);
                   std::swap(tiles[i], tiles[j]);
               }
               break;
           }
           //else {
           
            //if(i != j)
               // printf("Swap(%d %d)\n", i, j);
                //printf("%d %d %d %d\n", j, i, tiles[j][i].rank, counts[tiles[j][i].rank]);
                //counts[tiles[j][i].rank]++;
                //std::swap(tiles[j], tiles[i]);
              //  break;
            //}
        }
        //leader_ranks[i] = tiles[i][i].rank;
    }
    printf("\n");
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
    
}

bool TWOD_Staggered_New1() {
    //printf("2D-Staggered-New1\n");
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rank = ((i / rowgrp_nranks) * rowgrp_nranks) + (j % rowgrp_nranks);
        }
    }   
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
    
}

bool TWOD_Staggered_New() {
    
    std::vector<std::vector<int>> rowgrp_ranks;	
    rowgrp_ranks.resize(colgrp_nranks);	
    for(int i = 0; i < p; i++) {	
        int j = i / rowgrp_nranks;	
        rowgrp_ranks[j].push_back(i);	
    }
    /*
    for(int i = 0; i < colgrp_nranks; i++) {	
        for(int j = 0; j < rowgrp_nranks; j++) {    	
            printf("%d ", rowgrp_ranks[i][j]);	
        }	
        printf("\n");	
    }
    */
    //tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
    //printf("\n");	
    std::vector<int> offsets(rowgrp_nranks);
    for(int i = 0; i < nrowgrps; i++) {
        //int ii = (i % colgrp_nranks) * rowgrp_nranks;
        int row = ((i % colgrp_nranks) * rowgrp_nranks)/rowgrp_nranks;
        //int l = 
        //int iii = 
        
        //tile.rank = 
        //printf("%d %d %d %d\n", i, ii, row, i / colgrp_nranks);
        for(int j = 0; j < ncolgrps; j++) {
            int k = (i + j) % nrowgrps;
            int l = ((i / colgrp_nranks) + (j % rowgrp_nranks)) % rowgrp_nranks;
            tiles[i][k].rank = rowgrp_ranks[row][l];
           // printf("[%d %d %d] ", j, k, l);
        }
        //printf("\n");
        
    }
    
    /*	
    for(int ii = 0; ii < colgrp_nranks; ii++) {	
        auto& rg_ranks = rowgrp_ranks[ii];	
        if(std::find(rg_ranks.begin(), rg_ranks.end(), d) != rg_ranks.end()) {	
            row = ii;	
            break;	
        }	
    }	
    */	
    
    
   // for(int i = 0; i < nrowgrps; i++) {	    for(int i = 0; i < nrowgrps; i++) {
        //int d = tiles[i][i].rank;	
        //int d = i;	
        //int row = i/rowgrp_nranks;	
        //for(int r = 0; r < p; r++) {	
            
            //printf("<%d %d>\n", r, row);	
        //}	
        //printf("<%d %d %d>\n", d, row, row * rowgrp_nranks);	
      //  auto& rg_ranks = rowgrp_ranks[row];
        
    /*
    print("rank");
    exit(0);
    
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = i; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
            //printf("%d %d %d\n", i, j, i % colgrp_nranks);
        }
    }
    print("rank"); 
    exit(0);
    
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rg = i;
            tile.cg = j;
            tile.rank = (i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks);
            //printf("%d\n", not(p%2));
           //if(not (rank_ncolgrps%2))
           // if(not ((rank_nrowgrps%2) and (rank_ncolgrps%2)))
                tile.rank = ((((i % colgrp_nranks) * rowgrp_nranks) + (j % rowgrp_nranks)) + (((i/colgrp_nranks) * rowgrp_nranks) % p)) % p;
           // else 
             //   tile.rank = ((((i % colgrp_nranks) * rowgrp_nranks) + (j % rowgrp_nranks)) + (((i/rowgrp_nranks) * colgrp_nranks) % p)) % p;
               // tile.rank = (((i % colgrp_nranks) * rowgrp_nranks) + (j % rowgrp_nranks));
        }
    }
    */
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
}

int main(int argc, char **argv) {
    if(argc > 4) {
        printf("Usage: %s <tiling_ype> <numProcesses> <numThreads>\n", argv[0]);
        exit(0);
    }

    p = atoi(argv[2]);
    t = atoi(argv[3]); 
    
    
    printf("%d %d\n", p, t);
    if(not t) {
        nrowgrps = p;
        ncolgrps = p;
        rowgrp_nranks;
        colgrp_nranks;
        integer_factorize(p, rowgrp_nranks, colgrp_nranks);
        rank_nrowgrps = nrowgrps / colgrp_nranks;
        rank_ncolgrps = ncolgrps / rowgrp_nranks;
        tiles.resize(p);
        for(int i = 0; i < p; i++)
            tiles[i].resize(p);
        }
    else {
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
        
    }
    printf("p=%d, nrowgrps      x ncolgrps      = %d x %d\n", p, nrowgrps, ncolgrps);
    printf("p=%d, rank_nrowgrps x rank_ncolgrps = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
    printf("p=%d, rowgrp_nranks x colgrp_nranks = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);
    
    //printf("t=%d\n", t);


    
    bool ret = false;
    if(strcmp(argv[1], "2D-Staggered") == 0)
        ret = TWOD_Staggered();
    if(strcmp(argv[1], "2D-TStaggered") == 0)
        ret = TWOD_TStaggered();
    else if(strcmp(argv[1], "2D-Staggered-New") == 0)
        ret = TWOD_Staggered_New();
    else if(strcmp(argv[1], "2D-Staggered-New1") == 0)
        ret = TWOD_Staggered_New1();
    else 
        exit(0);
    
    return(ret);
}






