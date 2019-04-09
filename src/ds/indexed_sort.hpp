/*
 * indexed_sort.hpp: Indexed sort implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef INDEXED_SORT_HPP
#define INDEXED_SORT_HPP
template<typename Integer_Type>
std::vector<Integer_Type> sort_indices(const std::vector<Integer_Type>& v)  {
    Integer_Type sz = v.size();
    std::vector<Integer_Type> idx(sz);
    for(Integer_Type i = 0; i < sz; i++ )
        idx[i] = i;
    //std::iota(idx.begin(), idx.end(), 0); Because it might not be available
    std::sort(idx.begin(), idx.end(),
        [&v](Integer_Type i1, Integer_Type i2) {return v[i1] < v[i2];});
  return idx;
}

template<typename Integer_Type, typename Fractional_Type>
void indexed_sort(std::vector<Integer_Type>& v1, std::vector<Fractional_Type>& v2) {
    // Sort v2 based on v1 order
    std::vector<Integer_Type> idx = sort_indices(v1);
    std::sort(v1.begin(), v1.end());
    Fractional_Type max = *std::max_element(v2.begin(), v2.end());
    std::vector<Fractional_Type> temp(max + 1);
    Integer_Type i = 0;
    for(Integer_Type j: idx) {
        temp[v2[j]] = i;
        i++;
    }
    std::sort(v2.begin(), v2.end(),[&temp](Fractional_Type i1, Fractional_Type i2) {return temp[i1] < temp[i2];});           
}
#endif