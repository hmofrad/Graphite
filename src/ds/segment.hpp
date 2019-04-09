/*
 * vector.hpp: Vector implementaion
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef SEGMENT_HPP
#define SEGMENT_HPP

template<typename Weight, typename Integer_Type, typename Fractional_Type>
struct Segment {
    template<typename Weight_, typename Integer_Type_, typename Fractional_Type_>
    friend class Graph;
    
    template<typename Weight__, typename Integer_Type__, typename Fractional_Type__>
    friend class Matrix;
    
    template<typename Weight___, typename Integer_Type___, typename Fractional_Type___, typename Vertex_State, typename Vertex_Methods_Impl>
    friend class Vertex_Program;
    
    public:
        Segment();
        Segment(const Integer_Type nitems_, const int32_t socket_ids);
        ~Segment();
        void free();
        Integer_Type nitems;
        Fractional_Type* data;
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment() { }

template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::~Segment() { }

template<typename Weight, typename Integer_Type, typename Fractional_Type>
void Segment<Weight, Integer_Type, Fractional_Type>::free() {
    //printf("deleting?\n");
    uint64_t nbytes = 0;
    nbytes = nitems * sizeof(Fractional_Type);
    memset(data, 0, nbytes);    
    
    if(numa_available() != -1) {
        numa_free(data, nbytes);
    }
    else {
        if(munmap(data, nbytes) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }   
    }
}


template<typename Weight, typename Integer_Type, typename Fractional_Type>
Segment<Weight, Integer_Type, Fractional_Type>::Segment(const Integer_Type nitems_, const int32_t socket_ids) {
    nitems = nitems_;
    //vector_length = nitems.size();
    uint64_t nbytes = 0;
    
    nbytes = nitems * sizeof(Fractional_Type);
    if(numa_available() != -1) {
        data = (Fractional_Type*) numa_alloc_onnode(nbytes, socket_ids);
    }
    else {
        if((data = (Fractional_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
    }
    memset(data, 0, nbytes);    

}
#endif