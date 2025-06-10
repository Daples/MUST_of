#!/bin/bash

#SBATCH --job-name="rans-MUST"    # Name of the job for checking
#SBATCH --time=05:00:00                 # Wall clock time requested
#SBATCH --partition=compute-p2          # Which partition?
#SBATCH --account=research-ceg-gse      # Account to charge
#SBATCH --tasks=32                      # Number of tasks
#SBATCH --cpus-per-task=1               # Number of cpu per task
#SBATCH --mem-per-cpu=4000M                      # Ask for memory

### Run all the cases ###
nProcs=32		# Number of processors used for simpleFoam

module load 2024r1 openmpi
module load openfoam/2306

. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions

pwd

./Allclean
./Mesh $nProcs
./Allrun $nProcs

#------------------------------------------------------------------------------

touch must.foam

squeue -j $SLURM_JOBID
