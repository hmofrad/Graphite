/*
 * csc_spmv.hpp: CSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CSC_SPMV_HPP
#define CSC_SPMV_HPP

#include <chrono>

#include "pair.hpp" 
#include "io.cpp" 
#include "csc_base.hpp" 

class CSC {
    public:
        CSC() {};
        CSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~CSC() {};
        virtual void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair>* pairs = nullptr;
        struct CSC_BASE* csc = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x;
        std::vector<double> y;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        std::vector<char> rows;

        virtual void construct_filter();
        virtual void destruct_filter();
        void populate();
        void walk();
        void construct_vectors();
        void destruct_vectors();
        virtual void message();        
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};

void CSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    csc = new struct CSC_BASE(nedges, nvertices);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors();
    (void)spmv();
    v = y;
    //(void)checksum();
    //display();
    delete csc;
    csc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    csc = new struct CSC_BASE(nedges, nvertices);
    populate();
    total_size += csc->size;
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(rows[i] == 1)
            d[i] = v[i];
    }  
    //d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    for(uint32_t i = 0; i < niters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message();
        noperations += spmv();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "CSC SpMV");
    display();
    destruct_vectors();
    destruct_filter();  
    delete csc;
    csc = nullptr;
}

void CSC::construct_filter() {
    rows.resize(nvertices);
    for(auto &pair: *pairs)
        rows[pair.row] = 1;
}

void CSC::destruct_filter() {
    rows.clear();
    rows.shrink_to_fit();
}

void CSC::populate() {
    uint32_t* A  = (uint32_t*) csc->A;  // WEIGHT      
    uint32_t* IA = (uint32_t*) csc->IA; // ROW_IDX
    uint32_t* JA = (uint32_t*) csc->JA; // COL_PTR
    uint32_t ncols = csc-> ncols - 1;
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto& pair : *pairs) {
        while((j - 1) != pair.col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
    while((j + 1) < (ncols + 1)) {
        j++;
        JA[j] = JA[j - 1];
    }
}

void CSC::walk() {
    uint32_t* A  = (uint32_t*) csc->A;
    uint32_t* IA = (uint32_t*) csc->IA;
    uint32_t* JA = (uint32_t*) csc->JA;
    uint32_t ncols = csc->ncols;
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], j, A[i]);
        }
    }
}

void CSC::construct_vectors() {
    v.resize(nrows);
    x.resize(nrows, 1);
    y.resize(nrows);
    d.resize(nrows);
}

void CSC::destruct_vectors() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void CSC::message() {
    for(uint32_t i = 0; i < nrows; i++)
        x[i] = d[i] ? (v[i]/d[i]) : 0;   
}

uint64_t CSC::spmv() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) csc->A;
    uint32_t* IA = (uint32_t*) csc->IA;
    uint32_t* JA = (uint32_t*) csc->JA;
    uint32_t ncols = csc->ncols;
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
        }
    }
    return(noperations);
}

void CSC::update() {
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void CSC::space() {
    total_size += csc->size;
}

double CSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void CSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void CSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations << std::endl;
    std::cout << "Final value    : " << checksum() << std::endl;
}
#endif