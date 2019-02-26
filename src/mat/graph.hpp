/*
 * graph.hpp: Graph implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef GRAPH_HPP
#define GRAPH_HPP 
 
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>

#include "ds/triple.hpp"
#include "ds/compressed_column.hpp"
#include "mat/matrix.hpp"

/* Class template specialization for Weight
 * We think of two ways for processing graphs with empty weights:
 * 1) Representing the adjacency matrix with a matrix of type char,
 * 2) Infer the adjacency matrix from the csr or csc.
 * We picked the 2nd approach and use C macros to completely
 * remove the weight and adjacency matrix for empty weight graphs.
 * Thus, We have saved Graph_size * sizeof(Weight) in space because
 * representing the adjacency matrix even with char will cost us
 * Graph_size * sizeof(char). At scale, this means if e.g. a graph 
 * with 1B edges, we're saving 1B bytes of memory. Instead of using
 * C macrots, a better way is to create a base class and inherit it.
 * #define HAS_WEIGHT macro will be defined by compiler at compile time.
 */
template<typename Weight = char, typename Integer_Type = uint32_t, typename Fractional_Type = float>
class Graph {
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_, typename Vertex_state>
    friend class Vertex_Program;

    public:    
        Graph();
        ~Graph();
        
        void load(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_ = true, bool transpose_ = false, bool self_loops_ = true, bool acyclic_ = false,
            bool parallel_edges_ = true, Tiling_type tiling_type_ = _2DT_, Compression_type compression_type_ = _CSC_);
        void load_binary(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_, bool transpose_, bool self_loops_, bool acyclic_, bool parallel_edges_, Tiling_type tiling_type_, 
            Compression_type compression_type_);
        void load_text(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_,
            bool directed_, bool transpose_, bool self_loops_, bool acyclic_, bool parallel_edges_, Tiling_type tiling_type_, 
            Compression_type compression_type_);   
        void free();

    private:
        std::string filepath;
        Integer_Type nrows, ncols;
        uint64_t nedges;
        bool directed;
        bool transpose;
        bool self_loops;
        bool acyclic;
        bool parallel_edges;
        Matrix<Weight, Integer_Type, Fractional_Type>* A;
        void init_graph(std::string filepath_, Integer_Type nrows_, Integer_Type ncols_, 
               bool directed_, bool transpose_, bool self_loops_, bool acyclic_,
               bool parallel_edges_, Tiling_type tiling_type, Compression_type compression_type_);
        void parread_text();
        void parread_binary();
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::Graph() : A(nullptr) {};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Graph<Weight, Integer_Type, Fractional_Type>::~Graph(){};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::free() {
    A->del_compression();
    A->del_filter();
    A->free_tiling();    
    delete A;
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::init_graph(std::string filepath_, 
           Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,
           bool self_loops_, bool acyclic_, bool parallel_edges_, Tiling_type tiling_type_,
           Compression_type compression_type_) {
    filepath  = filepath_;
    nrows = nrows_ + 1; // In favor of vertex id 0
    ncols = ncols_ + 1; // In favor of vertex id 0
    nedges    = 0;
    directed  = directed_;
    transpose = transpose_;
    self_loops = self_loops_;
    acyclic = acyclic_;
    parallel_edges = parallel_edges_;
    
    uint32_t ntiles_ = 0;
    if((tiling_type_ == _1D_ROW_) or (tiling_type_ == _1D_COL_))
        ntiles_ = Env::nranks;
    else if(tiling_type_ == _2DN_)
        ntiles_ = (Env::nranks * Env::nthreads) * (Env::nranks * Env::nthreads);
        //ntiles_ = Env::nranks * Env::nranks * Env::nthreads;    
    else
        ntiles_ = Env::nranks * Env::nranks;
    
    //while(nrows % Env::nranks)
    while(nrows % (Env::nranks * Env::nthreads))
        nrows++;
    ncols = nrows;
    
    /*    
    if((tiling_type_ == _2D_) or (tiling_type_ == _2DT_))
        ntiles_ = Env::nranks * Env::nranks;
    else { 
        ntiles_ = Env::nranks;
        //printf("nrows=%d %d\n", nrows, nrows % Env::nranks);
        while(nrows % Env::nranks) {
            nrows++;
            ncols = nrows;
        }
        //printf("nrows=%d %d\n", nrows, nrows % Env::nranks);
    }
    */
    // Initialize matrix
    A = new Matrix<Weight, Integer_Type, Fractional_Type>(nrows, ncols, ntiles_, directed_, 
                                               transpose_, parallel_edges_, tiling_type_, compression_type_);
}

/* The read_text and read_binary functions can be called indivdually from the user program.
   The reasin we put this wrapper in between is to identify the file type inside the engine.*/
template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_,
        bool self_loops_, bool acyclic_, bool parallel_edges_, Tiling_type tiling_type_, 
        Compression_type compression_type_) {
    double t1, t2;
    t1 = Env::clock();
    int buffer_len = 100;
    char buffer[buffer_len];
    memset(buffer, '\n', buffer_len);
    
    int token_len = 10;
    char token[token_len];
    memset(token, '\n', token_len);
    
    FILE* fd = NULL;
    int len = 8 + filepath_.length() + 1; // file -b filepath\n
    char cmd[len];
    memset(cmd, '\0', len);
    sprintf(cmd, "file -b %s", filepath_.c_str());
    fd = popen(cmd, "r");
    while (fgets(buffer, buffer_len, fd) != NULL){ ; }
    pclose(fd);
    
    std::istringstream iss (buffer);
    iss >> token;
    const char* text = "ASCII";
    const char* data = "data";
    const char* data1 = "Hitachi";
    if(!strcmp(token, text)) {
        load_text(filepath_, nrows_, ncols_, directed_, transpose_, self_loops_, acyclic_, parallel_edges_, 
        tiling_type_, compression_type_);
    }
    else if(!strcmp(token, data) or !strcmp(token, data1)) {
        load_binary(filepath_, nrows_, ncols_, directed_, transpose_, self_loops_, acyclic_, parallel_edges_, 
        tiling_type_, compression_type_);
    }
    else {
        fprintf(stderr, "Undefined file type %s\n", token);
        Env::exit(1);
    }
    t2 = Env::clock();
    Env::print_time("Ingress", t2 - t1);
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_text(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_, bool self_loops_, bool acyclic_, bool parallel_edges_,
        Tiling_type tiling_type_, Compression_type compression_type_) {
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_,
               self_loops_, acyclic_, parallel_edges_,
               tiling_type_, compression_type_);
    // Read graph
    if(Env::is_master)
        printf("%s: Distributed read using %d ranks\n", filepath_.c_str(), Env::nranks);
    parread_text();
    // Initialize tiles
    A->init_tiles();
    // Filter the graph
    A->init_filtering();
    // Initialize threads
    A->init_threads();
    // Compress the graph
    A->init_compression();
    // Delete triples
    //A->del_triples();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::load_binary(std::string filepath_,
        Integer_Type nrows_, Integer_Type ncols_, bool directed_, bool transpose_, bool self_loops_, bool acyclic_, bool parallel_edges_,
        Tiling_type tiling_type_, Compression_type compression_type_) {
    // Initialize graph
    init_graph(filepath_, nrows_, ncols_, directed_, transpose_, 
               self_loops_, acyclic_, parallel_edges_, 
               tiling_type_, compression_type_);
    // Read graph
    if(Env::is_master)
        printf("%s: Distributed read using %d ranks\n", filepath_.c_str(), Env::nranks);
    parread_binary();
    // Initialize tiles
    A->init_tiles();
    
    if((tiling_type_ == _1D_ROW_) or (tiling_type_ == _1D_COL_)) {
        // Initialize threads 
        A->init_threads();              
        // Filter the graph
        A->init_filtering();
    }
    else {
        // Filter the graph
        A->init_filtering();        
        // Initialize threads
        //A->init_threads();
    }

    // Compress the graph
    A->init_compression();
    // Delete triples
    A->del_triples();
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::parread_text() {
    // Open graph file.
    std::ifstream fin(filepath.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Unable to open input file\n");
        Env::exit(1);
    }
    // Obtain filesize
    uint64_t filesize = 0, skip = 0, share = 0, offset = 0, endpos = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    // Skip comments
    std::string line;
    uint32_t position;  // Fallback position
    do {
        position = fin.tellg();
        std::getline(fin, line);
    } while ((line[0] == '#') || (line[0] == '%') || line.empty());
    /* Calculate the number of edges
     * We assume there's no empty line 
     * in the middle of the file
     */
    fin.clear();
    fin.seekg(position, std::ios_base::beg);
    while (std::getline(fin, line)) {
        if(line.empty()) {
            while(std::getline(fin, line)) {
                if(fin.eof() or !line.empty())
                    break;
            }
            if(fin.eof())
                break;
        }
        nedges++;
    }
    fin.clear();
    fin.seekg(position, std::ios_base::beg);

    share = nedges / Env::nranks;
    offset = share * Env::rank;
    endpos = (Env::rank == Env::nranks - 1) ? nedges : offset + share;
    while(skip < offset) {
        std::getline(fin, line);
        skip++;
    }

    uint64_t nedges_local = 0;
    uint64_t nedges_global = 0;
    struct Triple<Weight, Integer_Type> triple;
    std::istringstream iss;
    while (std::getline(fin, line) && !line.empty() && offset < endpos) {
        iss.clear();
        iss.str(line);
        #ifdef HAS_WEIGHT
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 3)
        #else
        if((std::count(line.cbegin(), line.cend(), ' ') + 1) != 2)
        #endif
        {
            fprintf(stderr, "read() failure \"%s\"\n", line.c_str());
            Env::exit(1);
        }
        nedges_local++;
        offset++;
        // Read weights if present
        #ifdef HAS_WEIGHT
        iss >> triple.row >> triple.col >> triple.weight;
        #else
        iss >> triple.row >> triple.col;
        #endif
        // Remove self-loops
        if (triple.row == triple.col) {
            if(not self_loops)
                continue;
        }
        // Remove cycles
        if(acyclic) {
            if(triple.col < triple.row)
                std::swap(triple.row, triple.col);
        }
        // Transpose
        if(transpose)
            std::swap(triple.row, triple.col);
        // Insert edge
        A->insert(triple);
        // Only for undirected graphs
        if(not directed) {
            std::swap(triple.row, triple.col);
            A->insert(triple);
        }
        // Print pipes
        if(Env::is_master) {
            if ((offset & ((1L << 26) - 1L)) == 0) {
                printf("|");
                fflush(stdout);
            }
        }
    }
    fin.close();
    
    assert(offset == endpos); 
    if(Env::rank == Env::nranks - 1)
        assert(filesize == endpos);
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges == nedges_global);
    MPI_Barrier(MPI_COMM_WORLD);
    if(Env::is_master)
        printf("\n%s: Read %lu edges\n", filepath.c_str(), nedges);
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Graph<Weight, Integer_Type, Fractional_Type>::parread_binary() {
    
    uint64_t nedges_local_r = 0;
    uint64_t nedges_global_r = 0;
    std::vector<uint64_t> nedges_local_t(Env::nthreads, 0);    
    std::vector<std::vector<struct Triple<Weight, Integer_Type>>> triples_for_threads(Env::nthreads);
    
    //Env::barrier();
    //Env::exit(0);
    
    #pragma omp parallel reduction(+:nedges_local_r)
    {
        // Open graph file.
        std::ifstream fin(filepath.c_str(), std::ios_base::binary);
        if(!fin.is_open()) {
            fprintf(stderr, "Unable to open input file\n");
            Env::exit(1); 
        }
        // Obtain filesize
        uint64_t filesize = 0, share = 0, offset = 0, endpos = 0;
        fin.seekg (0, std::ios_base::end);
        filesize = (uint64_t) fin.tellg();
        nedges = filesize / sizeof(Triple<Weight, Integer_Type>);
        //share = (filesize / (Env::nranks * Env::nthreads)) / sizeof(Triple<Weight, Integer_Type>) * sizeof(Triple<Weight, Integer_Type>);
        share = (filesize / Env::nranks) / sizeof(Triple<Weight, Integer_Type>) * sizeof(Triple<Weight, Integer_Type>);
        assert(share % sizeof(Triple<Weight, Integer_Type>) == 0);
        offset += share * Env::rank;
        endpos = (Env::rank == Env::nranks - 1) ? filesize : offset + share;
        fin.seekg(offset, std::ios_base::beg);
        
        //printf("rank=%d off=%lu end=%lu\n", Env::rank, offset, endpos);
        
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        uint64_t triples_range = endpos - offset;
        uint64_t share_t = (triples_range / nthreads) / sizeof(Triple<Weight, Integer_Type>) * sizeof(Triple<Weight, Integer_Type>);
        assert(share_t % sizeof(Triple<Weight, Integer_Type>) == 0);
        uint64_t start = offset + (share_t * tid);
        uint64_t end = start + share_t;
        end = (end > endpos) ? endpos : end;
        end = (tid == nthreads - 1) ? endpos : end;
        
        
        /*
        uint64_t ntriples = triples_range / sizeof(Triple<Weight, Integer_Type>);
        uint64_t ntriples_per_thread = ntriples/nthreads;
        uint64_t offset_t = 
        uint64_t start = offset + (tid * ntriples_per_thread * sizeof(Triple<Weight, Integer_Type>));
        uint64_t end = offset + (start + triples_range_per_thread);
        start = (start > endpos) ? (endpos):(start);
        end = (end > endpos) ? (endpos):(end);
        end = (tid == nthreads - 1) ? (endpos):(end);
        */

        fin.seekg(start, std::ios_base::beg);
        uint64_t position = fin.tellg();
        
        //if(Env::rank)
        //    printf("rank=%d tid=%d start=%lu end=%lu nt=%d pos=%lu\n", Env::rank, tid, start, end, (end - start)/sizeof(Triple<Weight, Integer_Type>) ,position);
        
        
        auto& nedges_local = nedges_local_t[tid];
        auto& triples = triples_for_threads[tid]; 
        
        
        struct Triple<Weight, Integer_Type> triple;
        while (start < end) {
            fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
            //printf("%d %d %d %d\n", Env::rank, tid, triple.row, triple.col);
            if(fin.gcount() != sizeof(Triple<Weight, Integer_Type>)) {
                fprintf(stderr, "read() failure");
                //fprintf(stderr, "read() failure rank=%d thread=%d nedges=%d start=%d end=%d pos=%d\n", Env::rank, tid, nedges_local, start, end, fin.tellg());
                Env::exit(1);
            }
            nedges_local++;
            start += sizeof(Triple<Weight, Integer_Type>);
            // Remove self-loops
            if (triple.row == triple.col) {
                if(not self_loops)
                    continue;
            }
            // Remove cycles
            if(acyclic) {
                if(triple.col < triple.row)
                    std::swap(triple.row, triple.col);
            }
            // Transpose
            if(transpose)
                std::swap(triple.row, triple.col);
            // Insert edge
            triples.push_back(triple);
            //A->insert(triple);
            // Only for undirected graphs        
            if(not directed) {
                std::swap(triple.row, triple.col);
                triples.push_back(triple);
               // A->insert(triple);
            }
            if(Env::is_master and !tid) {
                if ((start & ((1L << 26) - 1L)) == 0) {
                    printf("|");
                    fflush(stdout);
                }
            }
        }
        fin.close();
        assert(start == end);
        nedges_local_r += nedges_local;
        //printf("Rank=%d tid=%d start=%d end=%d nedges=%d\n", Env::rank, tid, start, end, nedges_local);

        
        

    }        
    
    /*
    uint64_t nedges_local = 0;
    uint64_t nedges_global = 0;
    struct Triple<Weight, Integer_Type> triple;
    while (offset < endpos) {
        fin.read(reinterpret_cast<char *>(&triple), sizeof(triple));
        if(fin.gcount() != sizeof(Triple<Weight, Integer_Type>)) {
            fprintf(stderr, "read() failure\n");
            Env::exit(1);
        }
        nedges_local++;
        offset += sizeof(Triple<Weight, Integer_Type>);
        // Remove self-loops
        if (triple.row == triple.col) {
            if(not self_loops)
                continue;
        }
        // Remove cycles
        if(acyclic) {
            if(triple.col < triple.row)
                std::swap(triple.row, triple.col);
        }
        // Transpose
        if(transpose)
            std::swap(triple.row, triple.col);
        // Insert edge
        A->insert(triple);
        // Only for undirected graphs        
        if(not directed) {
            std::swap(triple.row, triple.col);
            A->insert(triple);
        }
        if(Env::is_master) {
            if ((offset & ((1L << 26) - 1L)) == 0) {
                printf("|");
                fflush(stdout);
            }
        }
    }
    fin.close();
    
    
    */
    //assert(offset == endpos);
    
    
    MPI_Allreduce(&nedges_local_r, &nedges_global_r, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
    assert(nedges == nedges_global_r);
    //printf("%d %d %d\n", nedges, nedges_local_r, nedges_global_r);
    

    
    
    int s = 0;
    for(int i = 0; i < Env::nthreads; i++) {
        auto& triples = triples_for_threads[i]; 
        for(auto& triple: triples) {
            //A->test(triple);
            A->insert(triple);
        }
        //s += nedges_local_t[i];
    }
    //printf("sum=%d\n", s);
    //Env::barrier();
    
    
    Env::barrier();
    if(Env::is_master)
        printf("\n%s: Read %lu edges\n", filepath.c_str(), nedges);
    
}
#endif
