#!/bin/bash
#SBATCH --job-name=main
#SBATCH --output=main.out
#SBATCH --error=main.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --cluster=smp
#SBATCH --partition=high-mem

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

GRAPHS_PERFIX="/zfs1/cs3580_2017F/moh18/graph500/rmat"
GRAPHS=("rmat20.bin 1048576" "rmat21.bin 2097152" "rmat22.bin 4194304" "rmat23.bin 8388608" "rmat24.bin 16777216" "rmat25.bin 33554432" "rmat26.bin 67108864" "rmat27.bin 134217728" "rmat28.bin 268435456" "rmat29.bin 536870912" "rmat30.bin 1073741824")
PERF="/ihome/rmelhem/moh18/linux/tools/perf/./perf stat -B -e cycles,instructions,cs,faults,cache-references,cache-misses"
APPS=("0" "1" "2" "3" "4" "5")

for G in "${GRAPHS[@]}"
do
    for A in "${APPS[@]}"
    do
        CMD="./main ${A} ${GRAPHS_PERFIX}/${G} 20"
        echo "Command=${CMD}"
        ${CMD}
    done
done
exit;
