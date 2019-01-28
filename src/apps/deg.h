/*
 * deg.h: Degree benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DEG_H
#define DEG_H 

#include "vp/vertex_program.hpp"

using em = Empty;
#ifdef HAS_WEIGHT 
using wp = uint32_t; // Weight of type uint32_t 
#else
using wp = em;       // Weight of type empty (default)
#endif
using ip = uint32_t; // Integer precision for number of vertices
using fp = double;   // Fractional precision for precision of values.

struct Deg_State {
    ip degree = 0;
    ip get_state(){return(degree);};
    std::string print_state(){return("Degree=" + std::to_string(degree));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Deg_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, Deg_State> {
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, Deg_State>::Vertex_Program;
        
        virtual bool initializer(ip vid, Deg_State& state) {
            state.degree = 0; // Not necessary
            return(true);
        }

        virtual Fractional_Type messenger(Deg_State& state) {
            return(1);
        }
        
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2, const Fractional_Type& w) {
            y1 += (y2 * w);
        }
        
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2) {
            y1 += y2;
        }
        
        virtual bool applicator(Deg_State& state, const Fractional_Type& y) {
            state.degree = y;
            return(false);
        }   
};
#endif