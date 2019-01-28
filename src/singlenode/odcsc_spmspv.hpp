/*
 * odcsc.cpp: ODCSC SpMSpV implementation (LA3)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef ODCSC_SPMSPV_HPP
#define ODCSC_SPMSPV_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "odcsc_base.hpp"  

class ODCSC {  
    public:
        ODCSC() {};
        ODCSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~ODCSC() {};
        void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint64_t nedges_regulars = 0;
        uint64_t nedges_sources = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair> *pairs = nullptr;
        std::vector<struct Pair> *pairs_regulars = nullptr;
        std::vector<struct Pair> *pairs_sources = nullptr;
        struct ODCSC_BASE *odcsc_regulars = nullptr;
        struct ODCSC_BASE *odcsc_sources = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x_regulars;
        std::vector<double> x_sources;
        std::vector<double> x_sinks;
        std::vector<double> y;
        std::vector<double> y_regulars;
        std::vector<double> y_sources;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        
        uint32_t nnzrows_ = 0;
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzrows_regulars_ = 0;
        std::vector<char> rows_regulars;
        std::vector<uint32_t> rows_regulars_all;
        std::vector<uint32_t> rows_regulars_nnz;
        uint32_t nnzrows_sources_ = 0;
        std::vector<char> rows_sources;
        std::vector<uint32_t> rows_sources_all;
        std::vector<uint32_t> rows_sources_nnz;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        uint32_t nnzcols_regulars_ = 0;
        std::vector<char> cols_regulars;
        std::vector<uint32_t> cols_regulars_all;
        std::vector<uint32_t> cols_regulars_nnz;        
        uint32_t nnzcols_sinks_ = 0;
        std::vector<char> cols_sinks;
        std::vector<uint32_t> cols_sinks_all;
        std::vector<uint32_t> cols_sinks_nnz;
        std::vector<uint32_t> regulars_rows_nnzcols_to_cols_regulars;
        std::vector<uint32_t> sources_rows_nnzcols_to_cols_regulars;
        void construct_filter();
        void destruct_filter();
        
        void prepopulate();
        void populate();
        void walk();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
        void message_nnzcols();    
        void message_nnzcols_regulars_nnzrows();        
        uint64_t spmv_nnzrows_nnzcols();
        uint64_t spmv_nnzrows_regulars();
        uint64_t spmv_nnzrows_nnzcols_regulars();
        uint64_t spmv_nnzrows_regulars_nnzcols_regulars();
        void update();
        void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};
 
void ODCSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    pairs_regulars = new std::vector<struct Pair>;
    pairs_sources = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    construct_filter();
    prepopulate();
    odcsc_regulars = new struct ODCSC_BASE(nedges_regulars, nnzcols_);
    odcsc_sources = new struct ODCSC_BASE(nedges_sources, nnzcols_);
    populate();
    pairs_regulars->clear();
    pairs_regulars->shrink_to_fit();
    pairs_regulars = nullptr;
    pairs_sources->clear();
    pairs_sources->shrink_to_fit();
    pairs_sources = nullptr;
    //walk();
    construct_vectors_degree();
    (void)spmv_nnzrows_nnzcols();
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] =  y[i];
    //v = y;
    //(void)checksum();
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete odcsc_regulars;
    odcsc_regulars = nullptr;
    delete odcsc_sources;
    odcsc_sources = nullptr;

    // PageRank program
    pairs = new std::vector<struct Pair>;
    pairs_regulars = new std::vector<struct Pair>;
    pairs_sources = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    construct_filter();
    prepopulate();
    odcsc_regulars = new struct ODCSC_BASE(nedges_regulars, nnzcols_);
    odcsc_sources = new struct ODCSC_BASE(nedges_sources, nnzcols_);
    populate();
    space();
    pairs_regulars->clear();
    pairs_regulars->shrink_to_fit();
    pairs_regulars = nullptr;
    pairs_sources->clear();
    pairs_sources->shrink_to_fit();
    pairs_sources = nullptr;
    construct_vectors_pagerank();
    for(uint32_t i = 0; i < nrows; i++) {
        if(rows[i] == 1)
            d[i] = v[i];
    }
    //d = v;
    std::fill(v.begin(), v.end(), alpha);
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    /*
    for(uint32_t i = 1; i < niters; i++) {
        std::fill(x_regulars.begin(), x_regulars.end(), 0);
        std::fill(x_sources.begin(), x_sources.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzrows_nnzcols();
        update();
    }
    */
    if(niters == 1)
    {
        std::fill(x_regulars.begin(), x_regulars.end(), 0);
        std::fill(x_sources.begin(), x_sources.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzrows_nnzcols();
        update();
    }
    else
    {
        std::fill(x_regulars.begin(), x_regulars.end(), 0);
        std::fill(x_sources.begin(), x_sources.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzrows_regulars();
        update();
        for(uint32_t i = 1; i < niters - 1; i++)
        {
            std::fill(x_regulars.begin(), x_regulars.end(), 0);
            std::fill(y.begin(), y.end(), 0);
            message_nnzcols_regulars_nnzrows();
            noperations += spmv_nnzrows_regulars_nnzcols_regulars();
            update();
        }
        std::fill(x_regulars.begin(), x_regulars.end(), 0);
        std::fill(x_sources.begin(),  x_sources.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols_regulars_nnzrows();
        noperations += spmv_nnzrows_nnzcols_regulars();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "ODCSC SpMSpV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete odcsc_regulars;
    odcsc_regulars = nullptr;
    delete odcsc_sources;
    odcsc_sources = nullptr;
}


void ODCSC::construct_filter() {
    nnzrows_ = 0;
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto& pair:* pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;        
    }
    nnzrows_regulars_ = 0;
    rows_all.resize(nvertices);
    rows_regulars_all.resize(nvertices);
    rows_sources.resize(nvertices);
    nnzrows_sources_ = 0;
    rows_regulars.resize(nvertices);    
    rows_sources_all.resize(nvertices);    
    nnzcols_regulars_ = 0;
    cols_regulars.resize(nvertices);
    cols_regulars_all.resize(nvertices);    
    nnzcols_sinks_ = 0;
    cols_sinks.resize(nvertices);
    cols_sinks_all.resize(nvertices);
    for(uint32_t i = 0; i < nvertices; i++) {
        if(rows[i] == 1) {
            rows_nnz.push_back(i);
            rows_all[i] = nnzrows_;
            nnzrows_++;
            if(cols[i] == 0) {
                rows_sources[i] = 1;
                rows_sources_nnz.push_back(i);
                rows_sources_all[i] = nnzrows_sources_;
                nnzrows_sources_++;
            }
            if(cols[i] == 1) {
                rows_regulars[i] = 1;
                rows_regulars_nnz.push_back(i);
                rows_regulars_all[i] = nnzrows_regulars_;
                nnzrows_regulars_++;   
            }
        }
        if(cols[i] == 1) {
            nnzcols_++;
            if(rows[i] == 0) {
                cols_sinks[i] = 1;
                cols_sinks_nnz.push_back(i);
                cols_sinks_all[i] = nnzcols_sinks_;
                nnzcols_sinks_++;
            }
            if(rows[i] == 1) {
                cols_regulars[i] = 1;
                cols_regulars_nnz.push_back(i);
                cols_regulars_all[i] = nnzcols_regulars_;
                nnzcols_regulars_++;
            }
        }
    }
}

void ODCSC::destruct_filter() {
    nnzrows_ = 0;
    rows.clear();
    rows.shrink_to_fit();
    rows_all.clear();
    rows_all.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit(); 
    nnzrows_regulars_ = 0;
    rows_regulars.clear();
    rows_regulars.shrink_to_fit();
    rows_regulars_all.clear();
    rows_regulars_all.shrink_to_fit();
    rows_regulars_nnz.clear();
    rows_regulars_nnz.shrink_to_fit();
    nnzrows_sources_ = 0;
    rows_sources.clear();
    rows_sources.shrink_to_fit();
    rows_sources_all.clear();
    rows_sources_all.shrink_to_fit();
    rows_sources_nnz.clear();
    rows_sources_nnz.shrink_to_fit();
    nnzcols_ = 0;
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_regulars_ = 0;
    cols_regulars.clear();
    cols_regulars.shrink_to_fit();
    cols_regulars_all.clear();    
    cols_regulars_all.shrink_to_fit();    
    cols_regulars_nnz.clear();    
    cols_regulars_nnz.shrink_to_fit();    
    nnzcols_sinks_ = 0;
    cols_sinks.clear();
    cols_sinks.shrink_to_fit();
    cols_sinks_nnz.clear();
    cols_sinks_nnz.shrink_to_fit();
    cols_sinks_all.clear();
    cols_sinks_all.shrink_to_fit();
    regulars_rows_nnzcols_to_cols_regulars.clear();
    regulars_rows_nnzcols_to_cols_regulars.shrink_to_fit();
    sources_rows_nnzcols_to_cols_regulars.clear();
    sources_rows_nnzcols_to_cols_regulars.shrink_to_fit();
}

void ODCSC::prepopulate() {
    for(auto& pair : *pairs) {
        if(rows_regulars[pair.row])
            pairs_regulars->push_back(pair);
        else if(rows_sources[pair.row]) 
            pairs_sources->push_back(pair);
        else {
            printf("Invalid edge\n");
            exit(0);
        }
    }
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    nedges_regulars = pairs_regulars->size();
    nedges_sources = pairs_sources->size();
}
 
void ODCSC::populate() {
    // Regulars
    column_sort(pairs_regulars);
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES; // ENTRIES  
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA; // COL_PTR
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC; // COL_IDX
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    auto& p = pairs_regulars->front();
    JC[0] = p.col;
    for(auto& pair: *pairs_regulars) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                 
        ENTRIES[i].idx = pair.row;
        ENTRIES[i].global_idx = rows_all[pair.row];//rows_regulars_all[pair.row];
        ENTRIES[i].weight = 1;        
        JA[j]++;
        i++;
    }
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        if((JA[j + 1] - JA[j]) > 0) {
            if(cols_regulars[JC[j]])
                regulars_rows_nnzcols_to_cols_regulars.push_back(j);
        }
    }
    // Sources
    column_sort(pairs_sources);
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES; // ENTRIES  
    JA = (uint32_t *) odcsc_sources->JA; // COL_PTR
    JC = (uint32_t *) odcsc_sources->JC; // COL_IDX
    i = 0;
    j = 1;
    JA[0] = 0;
    p = pairs_sources->front();
    JC[0] = p.col;
    for(auto& pair: *pairs_sources) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                 
        ENTRIES[i].idx = pair.row;
        ENTRIES[i].global_idx = rows_all[pair.row];//rows_sources_all[pair.row];
        ENTRIES[i].weight = 1;        
        JA[j]++;
        i++;
    }
    for(uint32_t j = 0; j < nnzcols; j++) {
        if((JA[j + 1] - JA[j]) > 0) {
            if(cols_regulars[JC[j]])
                sources_rows_nnzcols_to_cols_regulars.push_back(j);
        }
    }
}    

void ODCSC::walk() {
    // Regulars
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            printf("    i=%d,%d, j=%d, value=%d\n", entry.idx, entry.global_idx, JC[j], entry.weight);
        }
    }
    // Sources
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES;
    JA = (uint32_t*) odcsc_sources->JA;
    JC = (uint32_t*) odcsc_sources->JC;
    nnzcols =  odcsc_sources->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            printf("    i=%d,%d, j=%d, value=%d\n", entry.idx, entry.global_idx, JC[j], entry.weight);
        }
    }
}   


void ODCSC::construct_vectors_degree() {
    v.resize(nrows);
    x_regulars.resize(nnzcols_, 1);
    x_sources.resize(nnzcols_, 1);
    y.resize(nnzrows_);
}

void ODCSC::destruct_vectors_degree() {
    x_regulars.clear();
    x_regulars.shrink_to_fit();
    x_sources.clear();
    x_sources.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void ODCSC::construct_vectors_pagerank() {
    x_regulars.resize(nnzcols_);
    x_sources.resize(nnzcols_);
    y.resize(nnzrows_);
    d.resize(nrows);
}

void ODCSC::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x_regulars.clear();
    x_regulars.shrink_to_fit();
    x_sources.clear();
    x_sources.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void ODCSC::message_nnzcols() {
    uint32_t* JC_R = (uint32_t*) odcsc_regulars->JC;
    uint32_t* JC_S = (uint32_t*) odcsc_sources->JC;
    uint32_t nnzcols = odcsc_regulars->nnzcols;
    for(uint32_t i = 0; i < nnzcols; i++) {
        if(d[JC_R[i]])
            x_regulars[i] = v[JC_R[i]]/d[JC_R[i]];
        if(d[JC_S[i]]) 
            x_sources[i] = v[JC_S[i]]/d[JC_S[i]];
    }
}

void ODCSC::message_nnzcols_regulars_nnzrows() {
    uint32_t* JC_R = (uint32_t*) odcsc_regulars->JC;
    uint32_t* JC_S = (uint32_t*) odcsc_sources->JC;
    for(uint32_t i : regulars_rows_nnzcols_to_cols_regulars) {
        if(d[JC_R[i]])
            x_regulars[i] = v[JC_R[i]]/d[JC_R[i]];
    }
    for(uint32_t i : sources_rows_nnzcols_to_cols_regulars) {
        if(d[JC_S[i]]) 
            x_sources[i] = v[JC_S[i]]/d[JC_S[i]];
    }   
}

uint64_t ODCSC::spmv_nnzrows_nnzcols() {
    // Regulars
    uint64_t noperations = 0;
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_regulars[j]);
            noperations++;
        }
    }
    // Sources
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES;
    JA = (uint32_t*) odcsc_sources->JA;
    JC = (uint32_t*) odcsc_sources->JC;
    nnzcols =  odcsc_sources->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_sources[j]);
            noperations++;
        }
    }
    return(noperations);
}

uint64_t ODCSC::spmv_nnzrows_regulars() {
    // Regulars
    uint64_t noperations = 0;
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    uint32_t nnzcols =  odcsc_regulars->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_regulars[j]);
            noperations++;
        }
    }
    return(noperations);
}

uint64_t ODCSC::spmv_nnzrows_regulars_nnzcols_regulars() {
    // Regulars
    uint64_t noperations = 0;
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    for(uint32_t j: regulars_rows_nnzcols_to_cols_regulars) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_regulars[j]);
            noperations++;
        }
    }
    return(noperations);
}

uint64_t ODCSC::spmv_nnzrows_nnzcols_regulars() {
    // Regulars
    uint64_t noperations = 0;
    CSCEntry* ENTRIES  = (CSCEntry*) odcsc_regulars->ENTRIES;
    uint32_t* JA = (uint32_t*) odcsc_regulars->JA;
    uint32_t* JC = (uint32_t*) odcsc_regulars->JC;
    for(uint32_t j: regulars_rows_nnzcols_to_cols_regulars) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_regulars[j]);
            noperations++;
        }
    }
    // Sources
    ENTRIES  = (CSCEntry*) odcsc_sources->ENTRIES;
    JA = (uint32_t*) odcsc_sources->JA;
    JC = (uint32_t*) odcsc_sources->JC;
    for(uint32_t j: sources_rows_nnzcols_to_cols_regulars) {
        for (uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            auto& entry = ENTRIES[i];
            y[entry.global_idx] += (entry.weight * x_sources[j]);
            noperations++;
        }
    }
    return(noperations);
}

void ODCSC::update() {
    for(uint32_t i = 0; i < nnzrows_; i++)
        v[rows_nnz[i]] = alpha + (1.0 - alpha) * y[i];
}

void ODCSC::space() {
    total_size += odcsc_regulars->size + odcsc_sources->size;
    total_size += sizeof(uint32_t) * regulars_rows_nnzcols_to_cols_regulars.size();
    total_size += sizeof(uint32_t) * sources_rows_nnzcols_to_cols_regulars.size();
    total_size += sizeof(uint32_t) * rows_nnz.size();
}

double ODCSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void ODCSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void ODCSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations << std::endl;
    std::cout << "Final value    : " << checksum() << std::endl;
}
#endif