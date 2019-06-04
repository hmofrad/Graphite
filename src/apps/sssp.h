/* 
 * bfs.h: Breadth First Search (BFS) benchmark helper
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef SSSP_H
#define SSSP_H

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

struct SSSP_State {
    ip distance = INF;
    ip get_state(){return(distance);};
    std::string print_state(){return((distance == INF) ? ("Distance=INF") : ("Distance=" + std::to_string(distance)));};
};

class SSSP_Methods_Impl {
  public:
    ip root = 0;
    inline void set_root(ip root_) { 
        root = root_; 
    };
    inline bool initializer(ip vid, SSSP_State& state) {
        if(vid == root) {
            state.distance = 0;
            return(true);
        }
        else {
            state.distance = INF;
            return(false);
        }
    }
    inline bool initializer(ip vid, SSSP_State& state, const State& other) {
        return(true);
    }
    inline fp messenger(SSSP_State& state) {
        return(state.distance);
    }
    inline void combiner(fp& y1, const fp& y2, const fp& w) {
        fp tmp = y2 + w;
        y1 = (y1 < tmp) ? y1 : tmp;
    }
    inline void combiner(fp& y1, const fp& y2) {
        y1 = (y1 < y2) ? y1 : y2;
    }
    inline bool applicator(SSSP_State& state) {
        return(false);
    }   
    inline bool applicator(SSSP_State& state, const fp& y) {
        fp tmp = state.distance;
        #ifdef HAS_WEIGHT
        state.distance = (y < state.distance) ? y : state.distance;
        #else
        state.distance = (y < state.distance) ? y + 1 : state.distance;
        #endif
        return(tmp != state.distance);
    }  
    inline bool applicator(SSSP_State& state, const fp& y, const ip iteration) {
        return(false); 
    }
    inline fp infinity() {
        return(INF);
    }
};



template<typename Weight, typename Integer_Type, typename Fractional_Type>
class SSSP_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, SSSP_State, SSSP_Methods_Impl> {
    public: 
        Integer_Type root = 0;
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, SSSP_State, SSSP_Methods_Impl>::Vertex_Program;        

        void set_root1(ip root_) { 
            root = root_; 
        };

        virtual bool initializer(Integer_Type vid, SSSP_State& state) {
            if(vid == root) {
                state.distance = 0;
                return(true);
            }
            else {
                state.distance = INF; // Not necessary
                return(false);
            }
        }
        
        virtual Fractional_Type messenger(SSSP_State& state) {
            return(state.distance);
        }

        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2, const Fractional_Type& w) {
            Fractional_Type tmp = y2 + w;
            y1 = (y1 < tmp) ? y1 : tmp;
        }
            
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2) {
            y1 = (y1 < y2) ? y1 : y2;
        }

        virtual bool applicator(SSSP_State& state, const Fractional_Type& y) {
            Fractional_Type tmp = state.distance;
            #ifdef HAS_WEIGHT
            state.distance = (y < state.distance) ? y : state.distance;
            #else
            state.distance = (y < state.distance) ? y + 1 : state.distance;
            #endif
            return(tmp != state.distance);
        }   
        
        virtual Fractional_Type infinity() {
            return(INF);
        }


};
#endif
