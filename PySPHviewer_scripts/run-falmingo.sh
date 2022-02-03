#!/bin/bash -l
#SBATCH --ntasks 256 # The number of cores you need...
#SBATCH --array=20-25
#SBATCH --cpus-per-task=2
#SBATCH -J FLAMINGO-ANI #Give it something meaningful.
#SBATCH -o logs/output_flamingo_ani.%J.%A.%a.out
#SBATCH -e logs/output_flamingo_ani.%J.%A.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 10:00:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma/home/dp004/dc-rope1/cosma7/Animations/codes/aXa_animations/PySPHviewer_scripts

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
#mpirun -np 512 python dm_animate_flamingo.py $i 0
mpirun -np 256 python gas_animate_flamingo.py $i

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit