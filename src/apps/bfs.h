/* 
 * bfs.h: Breadth First Search (BFS) benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef BFS_H
#define BFS_H

#include "vp/vertex_program.hpp"

#define INF 2147483647

using em = Empty;
#ifdef HAS_WEIGHT 
using wp = uint32_t; // Weight of type uint32_t 
#else
using wp = em;       // Weight of type empty (default)
#endif
using ip = uint32_t; // Integer precision for number of vertices
using fp = uint32_t; // Fractional precision for precision of values.

struct BFS_State {
    ip parent = 0;
    ip hops = INF;
    ip vid = 0;
    ip get_state(){return(hops);};
    //std::string print_state(){return("Parent=" + std::to_string(parent) + ",Hops=" + std::to_string(hops));};
    std::string print_state(){return((hops == INF) ? ("Parent=" + std::to_string(parent) + ",Hops=INF")
                                                   : ("Parent=" + std::to_string(parent) + ",Hops=" + std::to_string(hops)));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class BFS_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, BFS_State> {
    public:  
        Integer_Type root = 0;
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, BFS_State>::Vertex_Program;  // inherit constructors
        virtual bool initializer(Integer_Type vid, BFS_State& state) {
            if(vid == root) {
                state.vid = vid;
                state.parent = vid;
                state.hops = 0;
                return(true);
            }
            else {
                state.vid = vid;
                state.hops = INF; // Not necessary
                return(false);
            }
        }
        
        virtual Fractional_Type messenger(BFS_State& state) {
            return(state.vid);
        }
        
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2, const Fractional_Type& w) {
            Fractional_Type tmp = y2 + w;
            y1 = (y1 < tmp) ? y1 : tmp;
        }

        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2) {
            y1 = (y1 < y2) ? y1 : y2;
        }
        
        virtual bool applicator(BFS_State& state, const Fractional_Type& y, Integer_Type iteration) {
            if(state.hops != INF)
                return(false); // already visited
            else {
                if(y != INF) {
                    state.hops = iteration + 1;
                    state.parent = y;
                    return(true);
                }
                else
                    return(false);
            }
        }
        
        virtual Fractional_Type infinity() {
            return(INF);
        }        
};
#endif