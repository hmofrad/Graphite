/*
 * types.hpp: Get MPI data type from a templated class 
    Borrowed from https://github.com/thu-pacman/GeminiGraph/blob/master/core/mpi.hpp 
 */
#ifndef TYPES_HPP
#define TYPES_HPP
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Types {
    public:
        static MPI_Datatype get_data_type();
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
MPI_Datatype Types<Weight, Integer_Type, Fractional_Type>::get_data_type() {
    if (std::is_same<Fractional_Type, char>::value) {
        return MPI_CHAR;
    }
    else if (std::is_same<Fractional_Type, unsigned char>::value) {
        return MPI_UNSIGNED_CHAR;
    }
    else if (std::is_same<Fractional_Type, int>::value) {
        return MPI_INT;
    }
    else if (std::is_same<Fractional_Type, unsigned int>::value) {
        return MPI_UNSIGNED;
    }
    else if (std::is_same<Fractional_Type, long>::value) {
        return MPI_UNSIGNED_LONG;
    }
    else if (std::is_same<Fractional_Type, unsigned long>::value) {
        return MPI_UNSIGNED_LONG;
    }
    else if (std::is_same<Fractional_Type, float>::value) {
        return MPI_FLOAT;
    }
    else if (std::is_same<Fractional_Type, double>::value) {
        return MPI_DOUBLE;
    }
    else {
        fprintf(stderr, "Type not supported\n");
        Env::exit(1);
    }   
}
#endif