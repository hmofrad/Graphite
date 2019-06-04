/*
 * deg.h: Degree benchmark helper
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef DEG_H
#define DEG_H 

#include "vp/vertex_program.hpp"

#define INF 0

using em = Empty;
#ifdef HAS_WEIGHT 
using wp = uint32_t; // Weight of type uint32_t 
#else
using wp = em;       // Weight of type empty (default)
#endif
using ip = uint32_t; // Integer precision for number of vertices
using fp = double;   // Fractional precision for precision of values.

struct Deg_State {
    public:
        ip degree = 0;
        inline ip get_state(){return(degree);};
        inline std::string print_state(){return("Degree=" + std::to_string(degree));};
        /*
        inline bool initializer(ip vid, Deg_State& state) {
            state.degree = 0;
            return(true);
        }
        inline bool initializer(ip vid, Deg_State& state, const State& other) {
            return(true);
        }
        inline fp messenger(Deg_State& state) {
            return(1);
        }
        inline void combiner(fp& y1, const fp& y2, const fp& w) {
            y1 += (y2 * w);
        }
        inline void combiner(fp& y1, const fp& y2) {
            y1 += y2;
        }
        inline bool applicator(Deg_State& state) {
            return(false);
        }   
        inline bool applicator(Deg_State& state, const fp& y) {
            state.degree = y;
            return(false);
        }  
        inline bool applicator(Deg_State& state, const fp& y, const ip iteration) {
            return(false); 
        }
        inline fp infinity() {
            return(INF);
        }
        */
};
//__attribute__((packed));

class Deg_Methods_Impl {
  public:
    ip root = 0;
    inline void set_root(ip root_) { 
        root = root_; 
    };
    inline bool initializer(ip vid, Deg_State& state) {
        state.degree = 0;
        return(true);
    }
    inline bool initializer(ip vid, Deg_State& state, const State& other) {
        return(true);
    }
    inline fp messenger(Deg_State& state) {
        return(1);
    }
    inline void combiner(fp& y1, const fp& y2, const fp& w) {
        y1 += (y2 * w);
    }
    inline void combiner(fp& y1, const fp& y2) {
        y1 += y2;
    }
    inline bool applicator(Deg_State& state) {
        return(false);
    }   
    inline bool applicator(Deg_State& state, const fp& y) {
        state.degree = y;
        return(false);
    }  
    inline bool applicator(Deg_State& state, const fp& y, const ip iteration) {
        return(false); 
    }
    inline fp infinity() {
        return(INF);
    }
};


template<typename Weight, typename Integer_Type, typename Fractional_Type>
class Deg_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, Deg_State, Deg_Methods_Impl> {
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, Deg_State, Deg_Methods_Impl>::Vertex_Program;
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
            
            //asm ("add %2, %0" 
            //   : "=r" (y1) : "0" (y1), "r" (y2) : "cc");
            
            y1 += y2;
        }
        
        virtual bool applicator(Deg_State& state) {
            return(false);
        }   
        
        virtual bool applicator(Deg_State& state, const Fractional_Type& y) {
            state.degree = y;
            return(false);
        }   
};
#endif
