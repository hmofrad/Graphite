/*
 * io.cpp: I/O implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef IO_CPP
#define IO_CPP

#include <fstream>

uint64_t read_binary(const std::string file_path, std::vector<struct Pair> *pairs, bool transpose = false) {
    // Open graph file.
    std::ifstream fin(file_path.c_str(), std::ios_base::binary);
    if(!fin.is_open()) {
        fprintf(stderr, "Unable to open input file");
        exit(1); 
    }
    
    // Obtain filesize
    uint64_t nedges = 0, filesize = 0, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    struct Pair pair;
    while (offset < filesize) {
        fin.read(reinterpret_cast<char *>(&pair), sizeof(struct Pair));
        
        if(fin.gcount() != sizeof(struct Pair)) {
            fprintf(stderr, "read() failure\n");
            exit(1);
        }        
        nedges++;
        offset += sizeof(struct Pair);

        if(transpose)
            std::swap(pair.row, pair.col);
        
        pairs->push_back(pair);
    }
    fin.close();
    if(offset != filesize) {
        fprintf(stderr, "read() failure\n");
        exit(1);
    }
    if(transpose)
        printf("[x]I/O for transpose %s is done: Read %lu edges\n", file_path.c_str(), nedges);
    else
        printf("[x]I/O for original  %s is done: Read %lu edges\n", file_path.c_str(), nedges);
    return(nedges);
}

#endif