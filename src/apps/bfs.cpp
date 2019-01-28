/*
 * bfs.cpp: Breadth First Search (BFS) benchmark main
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#include <iostream>
#include <unistd.h>
 
#include "mpi/env.hpp"
#include "mat/graph.hpp"

#include "bfs.h"

int main(int argc, char** argv) { 
    Env::init();    
    double time1 = Env::clock();   
    if(argc != 3 and argc != 4) {
        if(Env::is_master)
            std::cout << "\"Usage: " << argv[0] << " <file_path> <num_vertices> <root>\"" << std::endl;
        Env::exit(1);
    }
    std::string file_path = argv[1]; 
    ip num_vertices = std::atoi(argv[2]);
    ip root = (argc > 3) ? std::atoi(argv[3]) : 0;
    bool directed = false;
    bool transpose = false;
    bool self_loops = false;
    bool acyclic = false;
    bool parallel_edges = false;
    Tiling_type TT = _2DT_;
    // Only CSC is supported for nonstationary algorithms
    Compression_type CT = _TCSC_; 
    
    /* Breadth First Search (BFS) execution */
    bool stationary = false;
    // Engine requirement for nonstationary algorithms on directed graphs
    if(not stationary and directed)
        transpose = not transpose; 
    Graph<wp, ip, fp> G;    
    G.load(file_path, num_vertices, num_vertices, directed, transpose, self_loops, acyclic, parallel_edges, TT, CT);
    bool gather_depends_on_apply = false;
    bool apply_depends_on_iter  = true;
    Ordering_type OT = _ROW_;
    BFS_Program<wp, ip, fp> V(G, stationary, gather_depends_on_apply, apply_depends_on_iter, OT);   
    V.root = root;
    V.execute();
    V.checksum();
    V.display();
    V.free();
    G.free();

    double time2 = Env::clock();
    Env::print_time("Breadth First Search (BFS) end-to-end", time2 - time1);
    Env::finalize();
    return(0);
}