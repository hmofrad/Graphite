# Graphite
   Graphite HPC Graph Analytics System 
## Install
    make
## Unistall
    make clean
## Usage
    mpirun -np Nprocesses bin/./deg   Graph.[bin/txt]   Nvertices
    mpirun -np Nprocesses bin/./pr    Graph.[bin/txt]   Nvertices [Niterations]
    mpirun -np Nprocesses bin/./sssp  GraphW.[bin/txt]  Nvertices
    mpirun -np Nprocesses bin/./bfs   Graph.[bin/txt]   Nvertices
    mpirun -np Nprocesses bin/./cc    Graph.[bin/txt]   Nvertices
## Paper
Mohammad Hasanzadeh Mofrad, Rami Melhem, Yousuf Ahmad and Mohammad Hammoud. [“Graphite: A NUMA-aware HPC System for Graph Analytics based on a new MPI∗X Parallelism Model.”](http://www.vldb.org/pvldb/vol13/p783-mofrad.pdf) In proceedings of the Very Large Data Bases Conference (PVLDB), Tokyo, Japan, 2020.

## Contact
    Mohammad Hasanzadeh Mofrad
    m.hasanzadeh.mofrad@gmail.com
