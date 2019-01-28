/*
 * pair.hpp: Pair structure implementation
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef PAIR_HPP
#define PAIR_HPP

#include <algorithm>

struct Pair {
    uint32_t row;
    uint32_t col;
};

struct ColSort {
    bool operator()(const struct Pair& a, const struct Pair& b) {
        return((a.col == b.col) ? (a.row < b.row) : (a.col < b.col));
    }
};

void column_sort(std::vector<struct Pair>* pairs) {
    ColSort f_col;
    std::sort(pairs->begin(), pairs->end(), f_col);
}
#endif