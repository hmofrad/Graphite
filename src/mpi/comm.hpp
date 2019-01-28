/*
 * comm.hpp: MPI serialize/deserialize
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef COMM_HPP
#define COMM_HPP

template<typename Weight, typename Integer_Type, typename Fractional_Type>    
class Comm
{
    public:
        static void pack_adjacency(std::vector<Integer_Type> &size_vec, 
                    std::vector<std::vector<Integer_Type>> &data,
                    std::vector<Integer_Type> &outbox);
                    
        static void unpack_adjacency(std::vector<Integer_Type> &size_vec, 
                    std::vector<std::vector<Integer_Type>> &data,
                    std::vector<Integer_Type> &inbox);                    
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>  
void Comm<Weight, Integer_Type, Fractional_Type>::pack_adjacency(std::vector<Integer_Type> &size_vec, 
                    std::vector<std::vector<Integer_Type>> &data,
                    std::vector<Integer_Type> &outbox)
{
    Integer_Type size_nitems = size_vec.size();
    size_vec[0] = 0;
    for(Integer_Type i = 1; i < size_nitems; i++)
    {
        size_vec[i] = size_vec[i-1] + data[i-1].size();
    } 

    Integer_Type outbox_nitems = size_vec[size_nitems - 1];    
    outbox.resize(outbox_nitems);
    Integer_Type k = 0;
    for(Integer_Type i = 0; i < size_nitems - 1; i++)
    {
        for(auto j: data[i])
        {
            outbox[k] = j;
            k++;
        }
    } 
}

template<typename Weight, typename Integer_Type, typename Fractional_Type>  
void Comm<Weight, Integer_Type, Fractional_Type>::unpack_adjacency(std::vector<Integer_Type> &size_vec, 
                    std::vector<std::vector<Integer_Type>> &data,
                    std::vector<Integer_Type> &inbox)
{
    Integer_Type size_nitems = size_vec.size();
    Integer_Type inbox_nitems = size_vec[size_nitems - 1];
    inbox.resize(inbox_nitems);
    
    for(uint32_t i = 0; i < size_nitems - 1; i++)
    {
        for(uint32_t j = size_vec[i]; j < size_vec[i+1]; j++)
            data[i].push_back(inbox[j]);  
    }
}

#endif
