/*
 * tcsc.hpp: TCSC SpMSpV implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TCSC_SPMVSPV_HPP
#define TCSC_SPMSPV_HPP 
 
#include <chrono> 
 
#include "pair.hpp" 
#include "io.cpp" 
#include "tcsc_base.hpp" 

class TCSC {  
    public:
        TCSC() {};
        TCSC(const std::string file_path_, const uint32_t nvertices_, const uint32_t niters_) 
            : file_path(file_path_), nvertices(nvertices_), niters(niters_) {}
        ~TCSC() {};
        void run_pagerank();
    protected:
        std::string file_path = "\0";
        uint32_t nvertices = 0;
        uint32_t niters = 0;
        uint64_t nedges = 0;
        uint32_t nrows = 0;
        std::vector<struct Pair>* pairs = nullptr;
        struct TCSC_BASE* tcsc = nullptr;
        std::vector<double> v;
        std::vector<double> d;
        std::vector<double> x;
        std::vector<double> x_r;
        std::vector<double> y;
        double alpha = 0.15;
        uint64_t noperations = 0;
        uint64_t total_size = 0;
        
        uint32_t nnzrows_ = 0;
        std::vector<char> rows;
        std::vector<uint32_t> rows_all;
        std::vector<uint32_t> rows_nnz;
        uint32_t nnzrows_regulars_ = 0;
        std::vector<uint32_t> rows_regulars_all;
        std::vector<uint32_t> rows_regulars_nnz;
        uint32_t nnzrows_sources_ = 0;
        std::vector<char> rows_sources;
        std::vector<uint32_t> rows_sources_all;
        std::vector<uint32_t> rows_sources_nnz;
        uint32_t nnzcols_ = 0;
        std::vector<char> cols;
        uint32_t nnzcols_regulars_ = 0;
        std::vector<uint32_t> cols_regulars_all;
        std::vector<uint32_t> cols_regulars_nnz;
        uint32_t nnzcols_sinks_ = 0;
        std::vector<uint32_t> cols_sinks_all;
        std::vector<uint32_t> cols_sinks_nnz;
        void construct_filter();
        void destruct_filter();
        
        void populate();
        void walk();
        void construct_vectors_degree();
        void destruct_vectors_degree();
        void construct_vectors_pagerank();
        void destruct_vectors_pagerank();
        void message_nnzcols();        
        uint64_t spmv_nnzcols();
        void message_nnzcols_regular_rows();        
        uint64_t spmv_nnzcols_regular();
        uint64_t spmv_nnzcols_regular_regular_rows();
        uint64_t spmv_nnzcols_regular_rows();
        void update();
        void space();
        void display(uint64_t nums = 10);
        double checksum();
        void stats(double elapsed_time, std::string type);
};

void TCSC::run_pagerank() {
    // Degree program
    nrows = nvertices;
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs);
    column_sort(pairs);
    construct_filter();
    tcsc = new struct TCSC_BASE(nedges, nnzcols_, nnzrows_, nnzcols_regulars_);
    populate();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;  
    //walk();
    construct_vectors_degree();
    (void)spmv_nnzcols();
    uint32_t *IR = (uint32_t *) tcsc->IR;
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++)
        v[IR[i]] =  y[i];
    //(void)checksum();
    //display();
    destruct_vectors_degree();
    destruct_filter();
    delete tcsc;
    tcsc = nullptr;
    
    // PageRank program
    pairs = new std::vector<struct Pair>;
    nedges = read_binary(file_path, pairs, true);
    column_sort(pairs);
    construct_filter();
    tcsc = new struct TCSC_BASE(nedges, nnzcols_, nnzrows_, nnzcols_regulars_);
    populate();        
    space();
    pairs->clear();
    pairs->shrink_to_fit();
    pairs = nullptr;
    //walk();
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
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzcols();
        update();        
    }
    */
    if(niters == 1)
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzcols();
        update();
    }
    else
    {
        std::fill(x.begin(), x.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols();
        noperations += spmv_nnzcols_regular_rows();
        update();
        for(uint32_t i = 1; i < niters - 1; i++)
        {
            std::fill(x_r.begin(), x_r.end(), 0);
            std::fill(y.begin(), y.end(), 0);
            message_nnzcols_regular_rows();
            noperations += spmv_nnzcols_regular_regular_rows();
            update();
        }
        std::fill(x_r.begin(), x_r.end(), 0);
        std::fill(y.begin(), y.end(), 0);
        message_nnzcols_regular_rows();
        noperations += spmv_nnzcols_regular();
        update();
    }
    t2 = std::chrono::steady_clock::now();
    auto t  = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    stats(t, "TCSC SpMSpV");
    display();
    destruct_vectors_pagerank();
    destruct_filter();
    delete tcsc;
    tcsc = nullptr;
}

void TCSC::construct_filter() {
    nnzrows_ = 0;
    rows.resize(nvertices);
    nnzcols_ = 0;
    cols.resize(nvertices);
    for(auto &pair: *pairs) {
        rows[pair.row] = 1;
        cols[pair.col] = 1;        
    }
    rows_all.resize(nvertices);
    nnzrows_regulars_ = 0;
    rows_regulars_all.resize(nvertices);
    rows_sources.resize(nvertices);
    nnzrows_sources_ = 0;
    rows_sources_all.resize(nvertices);    
    nnzcols_regulars_ = 0;
    cols_regulars_all.resize(nvertices);
    nnzcols_sinks_ = 0;
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
                rows_regulars_nnz.push_back(i);
                rows_regulars_all[i] = nnzrows_regulars_;
                nnzrows_regulars_++;   
            }
        }
        if(cols[i] == 1) {
            nnzcols_++;
            if(rows[i] == 0) {
                cols_sinks_nnz.push_back(i);
                cols_sinks_all[i] = nnzcols_sinks_;
                nnzcols_sinks_++;
            }
            if(rows[i] == 1) {
                cols_regulars_nnz.push_back(i);
                cols_regulars_all[i] = nnzcols_regulars_;
                nnzcols_regulars_++;
            }
        }
    } 
}

void TCSC::destruct_filter() {
    rows.clear();
    rows.shrink_to_fit();
    nnzrows_ = 0;
    rows_all.clear();
    rows_all.shrink_to_fit();
    rows_nnz.clear();
    rows_nnz.shrink_to_fit(); 
    cols.clear();
    cols.shrink_to_fit();
    nnzcols_ = 0;
    
    rows_sources_nnz.clear();
    rows_sources_nnz.shrink_to_fit();
    rows_sources_all.clear();
    rows_sources_all.shrink_to_fit();
    nnzrows_sources_ = 0;
    rows_sources.clear();
    rows_sources.shrink_to_fit();            
    rows_regulars_nnz.clear();
    rows_regulars_nnz.shrink_to_fit();
    rows_regulars_all.clear();
    rows_regulars_all.shrink_to_fit();
    nnzrows_regulars_ = 0;
    
    cols_sinks_nnz.clear();
    cols_sinks_nnz.shrink_to_fit();
    cols_sinks_all.clear();
    cols_sinks_all.shrink_to_fit();
    nnzcols_sinks_ = 0;    
    cols_regulars_nnz.clear();
    cols_regulars_nnz.shrink_to_fit();
    cols_regulars_all.clear();
    cols_regulars_all.shrink_to_fit();
    nnzcols_regulars_ = 0;
}

void TCSC::populate() {
    uint32_t* A  = (uint32_t*) tcsc->A;  // WEIGHT      
    uint32_t* IA = (uint32_t*) tcsc->IA; // ROW_IDX
    uint32_t* JA = (uint32_t*) tcsc->JA; // COL_PTR
    uint32_t* JC = (uint32_t*) tcsc->JC; // COL_IDX
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    auto& p = pairs->front();
    JC[0] = p.col;
    for(auto& pair : *pairs) {
        if(JC[j - 1] != pair.col) {
            JC[j] = pair.col;
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = rows_all[pair.row];
        i++;
    }
    // Rows indices
    uint32_t* IR = (uint32_t*) tcsc->IR; // ROW_PTR
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++)
        IR[i] = rows_nnz[i];
    // Regular columns pointers/indices
    uint32_t* JA_REG_C = (uint32_t*) tcsc->JA_REG_C; // COL_PTR_REG_COL
    uint32_t* JC_REG_C = (uint32_t*) tcsc->JC_REG_C; // COL_IDX_REG_COL
    uint32_t nnzcols = tcsc->nnzcols;
    uint32_t k = 0;
    uint32_t l = 0;
    for(uint32_t j = 0; j < nnzcols; j++) {
        if(JC[j] ==  cols_regulars_nnz[k]) {
            JC_REG_C[k] = JC[j];
            k++;
            JA_REG_C[l] = JA[j];
            JA_REG_C[l + 1] = JA[j + 1];
            l += 2;
        }
    }
    // Moving source rows to the end of indices
    uint32_t m = 0;
    uint32_t n = 0;
    std::vector<uint32_t> r;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            if(rows_sources[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            if(m > n) {
                for(uint32_t p = 0; p < n; p++) {
                    for(uint32_t q = JA[j+1] - 1; q >= JA[j]; q--) {
                        if(rows_sources[IR[IA[q]]] != 1) {
                            std::swap(IA[r[p]], IA[q]);
                            std::swap(A[r[p]], A[q]);
                            break;
                        }
                        else {
                            if(r[p] == q)
                                break;
                        }
                    }
                }
            }
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
    }
    // NNZ columns pointers without source rows
    l = 0;   
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    uint32_t* JA_REG_R = (uint32_t*) tcsc->JA_REG_R; // COL_PTR_REG_ROW
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            if(rows_sources[IR[IA[i]]] == 1) {
                m = (JA[j+1] - JA[j]);
                r.push_back(i);
            }   
        }
        if(m > 0) {
            n = r.size();
            JA_REG_R[l] = JA[j];
            JA_REG_R[l + 1] = JA[j + 1] - n;            
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_R[l] = JA[j];
            JA_REG_R[l + 1] = JA[j + 1];
            l += 2;  
        }
    }
    // Regular columns pointers without source rows
    l = 0;       
    m = 0;
    n = 0;
    r.clear();
    r.shrink_to_fit(); 
    uint32_t* JA_REG_RC = (uint32_t*) tcsc->JA_REG_RC; // COL_PTR_REG_COL_REG_ROW
  uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_REG_C[k]; i < JA_REG_C[k + 1]; i++) {
            if(rows_sources[IR[IA[i]]] == 1) {
                m = (JA_REG_C[k+1] - JA_REG_C[k]);
                r.push_back(i);
            }
        }
        if(m > 0) {
            n = r.size();
            JA_REG_RC[l] = JA_REG_C[k];
            JA_REG_RC[l + 1] = JA_REG_C[k + 1] - n;            
            l += 2; 
            m = 0;
            n = 0;
            r.clear();
            r.shrink_to_fit();
        }
        else {
            JA_REG_RC[l] = JA_REG_C[k];
            JA_REG_RC[l + 1] = JA_REG_C[k + 1];
            l += 2;  
        }
    }
}

void TCSC::walk() {
    uint32_t* A  = (uint32_t*) tcsc->A;
    uint32_t* IA = (uint32_t*) tcsc->IA;
    uint32_t* JA = (uint32_t*) tcsc->JA;
    uint32_t* JC = (uint32_t*) tcsc->JC;
    uint32_t* IR = (uint32_t*) tcsc->IR;
    uint32_t nnzcols = tcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        printf("j=%d/%d\n", j, JC[j]);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("   IA[%d]=%d/%d\n", i, IA[i], rows_sources[IR[IA[i]]]);
        }
    }
}

void TCSC::construct_vectors_degree() {
    v.resize(nrows);
    x.resize(nnzcols_, 1);
    y.resize(nnzrows_);
}

void TCSC::destruct_vectors_degree() {
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
}

void TCSC::construct_vectors_pagerank() {
    x.resize(nnzcols_);
    x_r.resize(nnzcols_regulars_);
    y.resize(nnzrows_);
    d.resize(nrows);
}

void TCSC::destruct_vectors_pagerank() {
    v.clear();
    v.shrink_to_fit();
    x.clear();
    x.shrink_to_fit();
    x_r.clear();
    x_r.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    d.clear();
    d.shrink_to_fit();
}

void TCSC::message_nnzcols() {
    uint32_t* JC = (uint32_t*) tcsc->JC;
    uint32_t nnzcols = tcsc->nnzcols;
    for(uint32_t i = 0; i < nnzcols; i++)
        x[i] = d[JC[i]] ? (v[JC[i]]/d[JC[i]]) : 0;   
}

uint64_t TCSC::spmv_nnzcols() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) tcsc->A;
    uint32_t* IA = (uint32_t*) tcsc->IA;
    uint32_t* JA = (uint32_t*) tcsc->JA;
    uint32_t nnzcols = tcsc->nnzcols;
    for(uint32_t j = 0; j < nnzcols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
        }
    }
    return(noperations);
}

uint64_t TCSC::spmv_nnzcols_regular_rows() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) tcsc->A;
    uint32_t* IA = (uint32_t*) tcsc->IA;
    uint32_t* JC = (uint32_t*) tcsc->JC;
    uint32_t* JA_REG_R = (uint32_t*) tcsc->JA_REG_R;
    uint32_t nnzcols = tcsc->nnzcols;
    for(uint32_t j = 0, k = 0; j < nnzcols; j++, k = k + 2) {
        //printf("%d %d %d\n", j, JA_REG_R[k + 1] - JA_REG_R[k], JC[j]);
        for(uint32_t i = JA_REG_R[k]; i < JA_REG_R[k + 1]; i++) {
            y[IA[i]] += (A[i] * x[j]);
            noperations++;
        }
    }
    return(noperations);
}

void TCSC::message_nnzcols_regular_rows() {
    uint32_t *JC_REG_C = (uint32_t *) tcsc->JC_REG_C;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t i = 0; i < nnzcols_regulars; i++)
        x_r[i] = d[JC_REG_C[i]] ? (v[JC_REG_C[i]]/d[JC_REG_C[i]]) : 0;   
}

uint64_t TCSC::spmv_nnzcols_regular_regular_rows() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) tcsc->A;
    uint32_t* IA = (uint32_t*) tcsc->IA;
    uint32_t* JA_REG_RC = (uint32_t*) tcsc->JA_REG_RC;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_REG_RC[k]; i < JA_REG_RC[k + 1]; i++) {
            y[IA[i]] += (A[i] * x_r[j]);
            noperations++;
        }
    }
    return(noperations);
}

uint64_t TCSC::spmv_nnzcols_regular() {
    uint64_t noperations = 0;
    uint32_t* A  = (uint32_t*) tcsc->A;
    uint32_t* IA = (uint32_t*) tcsc->IA;
    uint32_t* JA_REG_C = (uint32_t*) tcsc->JA_REG_C;
    uint32_t nnzcols_regulars = tcsc->nnzcols_regulars;
    for(uint32_t j = 0, k = 0; j < nnzcols_regulars; j++, k = k + 2) {
        for(uint32_t i = JA_REG_C[k]; i < JA_REG_C[k + 1]; i++) {
            y[IA[i]] += (A[i] * x_r[j]);
            noperations++;
        }
    }
    return(noperations);
}

void TCSC::update() {
    uint32_t* IR = (uint32_t*) tcsc->IR;
    uint32_t nnzrows = tcsc->nnzrows;
    for(uint32_t i = 0; i < nnzrows; i++)
        v[IR[i]] = alpha + (1.0 - alpha) * y[i];
}

void TCSC::space() {
    total_size += tcsc->size;
}

double TCSC::checksum() {
    double value = 0.0;
    for(uint32_t i = 0; i < nrows; i++)
        value += v[i];
    return(value);
}

void TCSC::display(uint64_t nums) {
    for(uint32_t i = 0; i < nums; i++)
        std::cout << "V[" << i << "]=" << v[i] << std::endl;
}

void TCSC::stats(double time, std::string type) {
    std::cout << type << " kernel unit test stats:" << std::endl;
    std::cout << "Utilized Memory: " << total_size / 1e9 << " GB" << std::endl;
    std::cout << "Elapsed time   : " << time / 1e6 << " Sec" << std::endl;
    std::cout << "Num Operations : " << noperations << std::endl;
    std::cout << "Final value    : " << checksum() << std::endl;
    std::cout << "nnzrows= " << nnzrows_ << " nnzrows_regulars= " << nnzrows_regulars_ << " nnzrows_sources= " << nnzrows_sources_ << std::endl;
    std::cout << "nnzcols= " << nnzcols_ << " nnzcols_regulars= " << nnzcols_regulars_ << " nnzcols_sinks= " << nnzcols_sinks_ << std::endl;
}
#endif