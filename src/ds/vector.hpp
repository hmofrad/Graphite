/*
 * vector.hpp: Vector implementaion
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Vector {
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Matrix;
    
    template<typename Weight___, typename Integer_Type___, typename Fractional_Type___, typename Vertex_State, typename Vertex_Methods_Impl>
    friend class Vertex_Program;
    
    public:
        Vector();
        Vector(const std::vector<Integer_Type> nitems_, const std::vector<Integer_Type> socket_ids);
        ~Vector();
        std::vector<Integer_Type> nitems;
        Fractional_Type** data;
        std::vector<bool> allocated;
        Integer_Type vector_length;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector() {};


template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::~Vector() {
    uint64_t nbytes = 0;
    if(numa_available() != -1) {
        for(uint32_t i = 0; i < vector_length; i++) {
            nbytes = nitems[i] * sizeof(Fractional_Type);
            memset(data[i], 0, nbytes);    
            numa_free(data[i], nbytes);
        }
        nbytes = vector_length * sizeof(Fractional_Type*);
        memset(data, 0, nbytes);    
        numa_free(data, nbytes);
    }
    else {
        for(uint32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Fractional_Type);
                memset(data[i], 0, nbytes);    
                if(munmap(data[i], nbytes) == -1) {
                    fprintf(stderr, "Error unmapping memory\n");
                    exit(1);
                }   
            }
        }
        nbytes = vector_length * sizeof(Fractional_Type*);
        memset(data, 0, nbytes);
        if(munmap(data, nbytes) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }   
    }
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Vector<Weight, Integer_Type, Fractional_Type>::Vector(const std::vector<Integer_Type> nitems_, const std::vector<Integer_Type> socket_ids) {
    nitems = nitems_;
    vector_length = nitems.size();
    uint64_t nbytes = 0;
    
    if(numa_available() != -1) {
        nbytes = vector_length * sizeof(Fractional_Type*);
        data = (Fractional_Type**) numa_alloc_onnode(nbytes, Env::socket_id);
        memset(data, 0, nbytes);
        
        for(uint32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Fractional_Type);
                data[i] = (Fractional_Type*) numa_alloc_onnode(nbytes, socket_ids[i]);
                memset(data[i], 0, nbytes);    
            }
            else {
                data[i] = nullptr;
            }
        }   
    }
    else {        
        nbytes = vector_length * sizeof(Fractional_Type*);
        if((data = (Fractional_Type**) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(data, 0, nbytes);
        
        for(uint32_t i = 0; i < vector_length; i++) {
            if(nitems[i]) {
                nbytes = nitems[i] * sizeof(Fractional_Type);
                if((data[i] = (Fractional_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                    fprintf(stderr, "Error mapping memory\n");
                    exit(1);
                }
                memset(data[i], 0, nbytes);
            }
            else {
                data[i] = nullptr;
            }
        }
    }
    
}
#endif