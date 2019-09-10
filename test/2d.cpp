/*
 * test.cpp: Tiling unit tests
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
//Compile and run using: g++ -o 2d 2d.cpp && ./2d 4 2

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

bool TOWD_Staggered() {
    printf("2D-Staggered\n");
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
        for(int j = i; j < ncolgrps; j++) {
            if(not (std::find(leader_ranks.begin(), leader_ranks.end(), tiles[j][i].rank) != leader_ranks.end())) {
                std::swap(tiles[j], tiles[i]);
                break;
            }
        }
        leader_ranks[i] = tiles[i][i].rank;
    }
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
}

bool TOWD_Staggered_New() {
    printf("2D-Staggered-New\n");
    for(int i = 0; i < nrowgrps; i++) {
        for(int j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j]; 
            tile.rg = i;
            tile.cg = j;
            tile.rank = ((((i % colgrp_nranks) * rowgrp_nranks) + (j % rowgrp_nranks)) + (((i/colgrp_nranks) * rowgrp_nranks) % p)) % p;
        }
    }
    print("rank"); 
    bool ret = check_diagonals();
    return(ret);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: %s <tiling_ype> <numProcesses> <numThreads>\n", argv[0]);
        exit(0);
    }
    p = atoi(argv[2]);
    t = atoi(argv[3]); 
    nrowgrps = p;
    ncolgrps = p;
    rowgrp_nranks;
    colgrp_nranks;
    integer_factorize(p, rowgrp_nranks, colgrp_nranks);
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;
    printf("p=%d, nrowgrps      x ncolgrps      = %d x %d\n", p, nrowgrps, ncolgrps);
    printf("p=%d, rank_nrowgrps x rank_ncolgrps = %d x %d\n", p, rank_nrowgrps, rank_ncolgrps);
    printf("p=%d, rowgrp_nranks x colgrp_nranks = %d x %d\n", p, rowgrp_nranks, colgrp_nranks);
    //printf("t=%d\n", t);

    tiles.resize(p);
    for(int i = 0; i < p; i++)
        tiles[i].resize(p);
    
    bool ret = false;
    if(strcmp(argv[1], "2D-Staggered") == 0)
        ret = TOWD_Staggered();
    else if(strcmp(argv[1], "2D-Staggered-New") == 0)
        ret = TOWD_Staggered_New();
    else 
        exit(0);
    
    return(ret);
}






