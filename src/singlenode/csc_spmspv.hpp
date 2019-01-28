/*
 * csc_spmspv.hpp: CSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CSC_SPMSPV_HPP
#define CSC_SPMSPV_HPP 

#include <chrono>

#include "pair.hpp" 
#include "io.cpp" 
#include "csc_spmspv.hpp"

class CSC_ : protected CSC {
    using CSC::CSC;
    public:
        virtual void run_pagerank();
    protected:
        virtual void message();
        virtual uint64_t spmv();
        virtual void update();
        virtual void space();
        
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzrows_ = 0;
        std::vector<char> cols;
        std::vector<uint32_t> cols_all;
        std::vector<uint32_t> cols_nnz;
        uint32_t nnzcols_ = 0;
        virtual void construct_filter();
        virtual void destruct_filter();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
};

void CSC_::run_pagerank() {    
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    csc = new struct CSC_BASE(nedges, nvertices);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors_degree();
    (void)spmv();
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] =  y[i];
    //(void)checksum();
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete csc;
    csc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    csc = new struct CSC_BASE(nedges, nvertices);
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
    stats(t, "CSC SpMSpV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete csc;
    csc = nullptr;
}

void CSC_::construct_filter() {
    nnzrows_ = 0;
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto& pair : *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;
    }
    rows_all.resize(nvertices);
    cols_all.resize(nvertices);
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = nnzrows_;
            nnzrows_++;
        }
        if(cols[i] == 1) {
            cols_nnz.push_back(i);
            cols_all[i] = nnzcols_;
            nnzcols_++;
        }
    }
}

void CSC_::destruct_filter() {
    rows.clear();
    rows.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit();
    rows_all.clear();
    rows_all.shrink_to_fit();
    nnzrows_ = 0;
    cols.clear();
    cols.shrink_to_fit();
    cols_nnz.clear();
    cols_nnz.shrink_to_fit();
    cols_all.clear();
    cols_all.shrink_to_fit();
    nnzcols_ = 0;
}

void CSC_::construct_vectors_degree() {
    v.resize(nrows);
    x.resize(nnzcols_, 1);
    y.resize(nnzrows_);
}

void CSC_::destruct_vectors_degree() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}


void CSC_::construct_vectors_pagerank() {
    x.resize(nnzcols_);
    y.resize(nnzrows_);
    d.resize(nrows);
}

void CSC_::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void CSC_::message() {
    for(uint32_t i = 0; i < nnzcols_; i++)
    {
        x[i] = d[cols_nnz[i]] ? (v[cols_nnz[i]]/d[cols_nnz[i]]) : 0;   
    }
}

uint64_t CSC_::spmv() {
    uint64_t num_operations = 0;
    uint32_t* A  = (uint32_t*) csc->A;
    uint32_t* IA = (uint32_t*) csc->IA;
    uint32_t* JA = (uint32_t*) csc->JA;
    uint32_t ncols = csc->ncols;
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[rows_all[IA[i]]] += (A[i] * x[cols_all[j]]);
            num_operations++;
        }
    }
    return(num_operations);
}

void CSC_::update() {
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void CSC_::space() {
    total_size += csc->size;
    total_size += (sizeof(uint32_t) * rows_all.size()) + (sizeof(uint32_t) * rows_nnz.size());
    total_size += (sizeof(uint32_t) * cols_all.size()) + (sizeof(uint32_t) * cols_nnz.size());
}
#endif