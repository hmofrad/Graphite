#!/bin/bash
#SBATCH --job-name=graphit
#SBATCH --output=graphit.out
#SBATCH --error=graphit.err
#SBATCH --ntasks=64
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
##SBATCH --mem=50000
##SBATCH --sockets-per-node=2
##SBATCH --ntasks-per-socket=8
#SBATCH --time=01:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem

## while true; do free -g; sleep 10; done
## grep 'Execution' la3.out
## strings graphtap.out | grep 'Execute'
## 

echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "************************************************"

module purge
module load gcc/5.4.0
module load intel
export I_MPI_FABRICS=shm:ofa

export SLURM_CPU_BIND="none"
export I_MPI_FALLBACK=0
export LD_LIBRARY_PATH=/ihome/rmelhem/moh18/boost/boost_1_67_0/stage/lib:/ihome/rmelhem/moh18/boost:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NP=$SLURM_NTASKS
MPI="mpirun -np"
APPS_PERFIX="bin"
APPS=("./pr" "./sssp" "./bfs" "./cc")
GRAPHS_PERFIX="/zfs1/cs3580_2017F/moh18/graph500/rmat"
GRAPHS=("rmat26_1.bin" "rmat27_1.bin" "rmat28.bin" "rmat29.bin" "rmat30.bin")
GRAPHS_W=("rmat26_1_w.bin0" "rmat27_1_w.bin0" "rmat28_w.bin" "rmat29_w.bin" "rmat30_w.bin")
GRAPHS_V=("67108863" "134217728" "268435456" "536870912" "1073741824")
SRC_V=("1554330" "1554330" "1" "1554332" "1")


if [ "${SLURM_NNODES}" = "4" ]
then
    I="0"
elif [ "$SLURM_NNODES" = "8" ]
then
    I="1"
elif [ "$SLURM_NNODES" = "16" ]
then
    I="2"
elif [ "$SLURM_NNODES" = "24" ]
then
    I="3"
elif [ "$SLURM_NNODES" = "32" ]
then
    I="4"
else
    echo "Invalid input graph"
    exit;     
fi
#GRAPH=${GRAPHS[${I}]}    
GRAPH_V=${GRAPHS_V[${I}]}
SRC=${SRC_V[${I}]}
ITERS=20

PERF="/ihome/rmelhem/moh18/linux/tools/perf/./perf stat -B -e cycles,instructions,cs,faults,cache-references,cache-misses"
NTIMES=("1");
#NTIMES=("1" "2" "3");
for A in "${APPS[@]}";
do
    for N in "${NTIMES[@]}";
    do
        if [ "${A}" = "./pr" ]
        then
            ITER_OR_SRC=${ITERS}
        else
            ITER_OR_SRC=${SRC}
        fi
        
        if [ "${A}" = "./sssp" ]
        then
            GRAPH="${GRAPHS_W[${I}]}"    
        else
            GRAPH="${GRAPHS[${I}]}"
        fi    
        
        CMD="${MPI} ${NP} ${APPS_PERFIX}/${A} ${GRAPHS_PERFIX}/${GRAPH} ${GRAPH_V} ${ITER_OR_SRC}"
        echo "Command=${CMD}"
        ${CMD}
    done
done

strings graphit.out | grep 'Execute time'
strings graphit.out | grep 'Iterations'
strings graphit.out | grep 'Value checksum'
strings graphit.out | grep 'Reachable vertices'
#strings graphtap.out | grep 'avg'
