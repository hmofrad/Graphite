/*
 * cc.h: Connected Component (CC) benchmark helper
 * (c) Mohammad Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef CC_H
#define CC_H

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

struct CC_State {
    ip label = 0;
    ip get_state(){return(label);};   
    std::string print_state(){return("Label=" + std::to_string(label));};
};


class CC_Methods_Impl {
  public:
    ip root = 0;
    inline void set_root(ip root_) { 
        root = root_; 
    };
    inline bool initializer(ip vid, CC_State& state) {
        state.label = vid;
        return(true);
    }
    inline bool initializer(ip vid, CC_State& state, const State& other) {
        return(true);
    }
    inline fp messenger(CC_State& state) {
        return(state.label);
    }
    inline void combiner(fp& y1, const fp& y2, const fp& w) {
        fp tmp = y2 + w;
        y1 = (y1 < tmp) ? y1 : tmp;
    }
    inline void combiner(fp& y1, const fp& y2) {
        y1 = (y1 < y2) ? y1 : y2;
    }
    inline bool applicator(CC_State& state) {
        return(false);
    }   
    inline bool applicator(CC_State& state, const fp& y) {
        fp tmp = state.label;
        state.label = (y < state.label) ? y : state.label;
        return(tmp != state.label);
    }  
    inline bool applicator(CC_State& state, const fp& y, const ip iteration) {
        return(false);
    }
    inline fp infinity() {
        return(INF);
    }
};



template<typename Weight, typename Integer_Type, typename Fractional_Type>
class CC_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, CC_State, CC_Methods_Impl> {
    public:  
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, CC_State, CC_Methods_Impl>::Vertex_Program;
        
};      
/*
template<typename Weight, typename Integer_Type, typename Fractional_Type>
class CC_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, CC_State> {
    public:  
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, CC_State>::Vertex_Program;
        virtual bool initializer(Integer_Type vid, CC_State& state) {
            state.label = vid;
            return(true);
        }
        
        virtual Fractional_Type messenger(CC_State& state) {
            return(state.label);
        }
        
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2, const Fractional_Type& w) {
            Fractional_Type tmp = y2 + w;
            y1 = (y1 < tmp) ? y1 : tmp;
        }

        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2) {
            y1 = (y1 < y2) ? y1 : y2;
        }

        virtual bool applicator(CC_State& state, const Fractional_Type& y) {
            Fractional_Type tmp = state.label;
            state.label = (y < state.label) ? y : state.label;
            return(tmp != state.label);
        }   
        
        virtual Fractional_Type infinity() {
            return(INF);
        }
};
*/
#endif