/*
 * dcsc.hpp: DCSC SpMV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DCSC_SPMV_HPP
#define DCSC_SPMV_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "dcsc_base.hpp" 
 
class DCSC {  
    public:
        DCSC() {};
        DCSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~DCSC() {};
        virtual void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair>* pairs = nullptr;
        struct DCSC_BASE* dcsc = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x;
        std::vector<double> y;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        std::vector<char> rows;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        void construct_filter();
        void destruct_filter();
        
        void populate();
        void walk();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
        virtual void message();        
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};

void DCSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    dcsc = new struct DCSC_BASE(nedges, nnzcols_);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors_degree();
    (void)spmv();
    v = y;
    //(void)checksum();
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete dcsc;
    dcsc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    dcsc = new struct DCSC_BASE(nedges, nnzcols_);
    populate();        
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    construct_vectors_pagerank(); 
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
    stats(t, "DCSC SpMV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete dcsc;
    dcsc = nullptr;
}

void DCSC::construct_filter() {
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    for(uint32_t i = 0; i < nvertices; i++) {
        if(cols[i] == 1)
            nnzcols_++;
    }
}

void DCSC::destruct_filter() {
    rows.clear();
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_ = 0;
}

void DCSC::populate() {
    uint32_t* A  = (uint32_t*) dcsc->A;  // WEIGHT      
    uint32_t* IA = (uint32_t*) dcsc->IA; // ROW_IDX
    uint32_t* JA = (uint32_t*) dcsc->JA; // COL_PTR
    uint32_t* JC = (uint32_t*) dcsc->JC; // COL_IDX
    uint32_t i = 0;
    uint32_t j = 1;
    uint32_t k = 1;
    JA[0] = 0;
    auto &p = pairs->front();
    JC[0] = p.col;
    for(auto &pair : *pairs) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = pair.row;
        i++;
    }
}

void DCSC::walk() {
    uint32_t* A  = (uint32_t*) dcsc->A;
    uint32_t* IA = (uint32_t*) dcsc->IA;
    uint32_t* JA = (uint32_t*) dcsc->JA;
    uint32_t* JC = (uint32_t*) dcsc->JC;
    uint32_t nnzcols = dcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], JC[j], A[i]);
        }
    }
}

void DCSC::construct_vectors_degree() {
    v.resize(nrows);
    x.resize(nnzcols_, 1);
    y.resize(nrows);
}

void DCSC::destruct_vectors_degree() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void DCSC::construct_vectors_pagerank() {
    x.resize(nnzcols_);
    y.resize(nrows);
    d.resize(nrows);
}

void DCSC::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void DCSC::message() {
    uint32_t* JC = (uint32_t*) dcsc->JC;
    uint32_t nnzcols = dcsc->nnzcols;
    for(uint32_t i = 0; i < nnzcols; i++)
        x[i] = d[JC[i]] ? (v[JC[i]]/d[JC[i]]) : 0;   
}

uint64_t DCSC::spmv() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) dcsc->A;
    uint32_t* IA = (uint32_t*) dcsc->IA;
    uint32_t* JA = (uint32_t*) dcsc->JA;
    uint32_t nnzcols = dcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
        }
    }
    return(noperations);
}

void DCSC::update() {
    for(uint32_t i = 0; i < nrows; i++)
        v[i] = alpha + (1.0 - alpha) * y[i];
}

void DCSC::space() {
    total_size += dcsc->size;
}

double DCSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void DCSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void DCSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations << std::endl;
    std::cout << "Final value    : " << checksum() << std::endl;
}
#endif