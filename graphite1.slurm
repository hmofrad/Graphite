#!/bin/bash
#SBATCH --job-name=graphite1
#SBATCH --output=graphite1.out
#SBATCH --error=graphite1.err
#SBATCH --ntasks=8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=150000
#SBATCH --time=01:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem
###SBATCH --nodelist=opa-n[120-123]

## while true; do free -g; sleep 10; done
## grep 'Execution' la3.out
## strings graphtap.out | grep 'Execute'
 

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
#APPS=("./pr")
APPS=("./pr" "./sssp" "./bfs" "./cc")

GRAPHS_PERFIX="/zfs1/cs3580_2017F/moh18/real"
GRAPHS=("usa-roads-1.bin" "live-journal-1.bin"  "protein-k-mer-1.bin" "uk-2005-1.bin" "it-2004-1.bin0" "twitter-2010-1.bin" "gsh-2015-host-1.bin0" "uk-2006-06-1.bin" "uk-union-1.bin")
GRAPHS_W=("usa-roads-1-w.bin0" "live-journal-1-w.bin0" "protein-k-mer-1-w.bin0" "uk-2005-1-w.bin0" "it-2004-1-w.bin0" "twitter-2010-1-w.bin0" "gsh-2015-host-1-w.bin0" "uk-2006-06-1-w.bin0" "uk-union-1-w.bin0")
GRAPHS_V=("23947348" "4847572"  "214005018" "39459926" "41291596" "41652231" "68660142"  "80644903" "133633041")
SRC_V=("200" "10010" "2" "21049533" "6767" "23934133" "901503" "48018932" "26927947")

if [ "${SLURM_NNODES}" = "2" ]
then
    II=("0" "1" "2");
elif [ "${SLURM_NNODES}" = "4" ]
then
    II=("3" "4");
elif [ "$SLURM_NNODES" = "8" ]
then
    II=("5" "6");
elif [ "$SLURM_NNODES" = "16" ]
then
    II=("7");
elif [ "$SLURM_NNODES" = "24" ]
then
    II=("8")
else
    echo "Invalid input graph"
    exit;     
fi


for I in "${II[@]}";
    do
    GRAPH_V=${GRAPHS_V[${I}]}
    SRC=${SRC_V[${I}]}
    ITERS=20
    NTIMES=("1")
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
done

strings graphite1.out | grep 'Execute time'
#strings graphite1.out | grep 'Iterations'
#strings graphite1.out | grep 'Value checksum'
#strings graphite1.out | grep 'Reachable vertices'

