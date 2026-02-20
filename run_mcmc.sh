#!/bin/bash
#SBATCH --job-name=<your job name>
#SBATCH --time=4-00:00
#SBATCH --partition=short,medium,long
#SBATCH --output=./projects/<your output file>
#SBATCH --error=./projects/<your log file>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mail-user=<your grid email>
#SBATCH --mail-type=ALL

YAML=./projects/hdstest/MCMC${SLURM_ARRAY_TASK_ID}.yaml # This is the path where the MCMC files must be. They must be numerated as MCMC1.yaml, MCMC2.yaml and so on. 

echo "Job started in `hostname` at `date`"

# NOTE: sometimes `source start_cocoa.sh` fails if many jobs are loaded simultaneously.
# This sleep prevents bugs from happening
# sleep $(( 10 + SLURM_ARRAY_TASK_ID*20 ))

cd ~/cocoa/Cocoa
conda init bash
conda activate cocoa
source start_cocoa.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# NOTE: some libraries may use these variables to control threading
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -n ${SLURM_NTASKS_PER_NODE} --mca btl tcp,self --bind-to core:overload-allowed --rank-by slot --map-by core cobaya-run ${YAML} -r

echo "Job ended at `date`"
