/*
 * pr.h: PageRank benchmark helper
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef PR_H
#define PR_H 

#include "deg.h"

fp tol = 1e-5;
fp alpha = 0.15;

struct PR_State : Deg_State {
    fp rank = alpha;
    fp get_state(){return(rank);};
    std::string print_state(){return("Rank=" + std::to_string(rank) + ",Degree=" + std::to_string(degree));};
};

template<typename Weight, typename Integer_Type, typename Fractional_Type>
class PR_Program : public Vertex_Program<Weight, Integer_Type, Fractional_Type, PR_State> {
    public: 
        using Vertex_Program<Weight, Integer_Type, Fractional_Type, PR_State>::Vertex_Program;
        virtual bool initializer(Integer_Type vid, PR_State& state, const State& other) {
            state.degree = ((const Deg_State&) other).degree;
            state.rank = alpha; //  Not necessary
            return(true);
        }

        virtual Fractional_Type messenger(PR_State &state) {
            return( (state.degree) ? (state.rank / state.degree) : 0 );
        }

        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2, const Fractional_Type& w) {
            y1 += (y2 * w);
        }
        
        virtual void combiner(Fractional_Type& y1, const Fractional_Type& y2) {
            y1 += y2;
        }
        
        virtual bool applicator(PR_State& state, const Fractional_Type& y) {
            Fractional_Type tmp = state.rank;
            state.rank = alpha + (1.0 - alpha) * y;
            return (fabs(state.rank - tmp) > tol);         
        }
};
#endif