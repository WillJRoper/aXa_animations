#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-380
#SBATCH --cpus-per-task=8
#SBATCH -J 12p5DMO-ani #Give it something meaningful.
#SBATCH -o logs/output_flythrough.%J.%A.%a.out
#SBATCH -e logs/output_flythrough.%J.%A.%a.err
#SBATCH -p cosma6 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma/home/dp004/dc-rope1/cosma7/Animations/codes/aXa_animations/PySPHviewer_scripts

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1 + 1000))

# Run the program
python grid_cubes.py $i
# python dm_animate.py $i
# python gas_density_animate.py $i
# python gas_temp_animate.py $i
python stars_animate.py $i

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit