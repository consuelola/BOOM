#!/bin/bash
# To start this job : qsub submitjob.sh
# Job description
#PBS -N submitjob
# Resources used : one core on one node
#PBS -l nodes=1:ppn=5
# 10 hour elapsed time
#PBS -l walltime=10:00
# give 8gb of memory for your job
# memory is for the all process in a job not per process
# vmem must be >= mem .
#PBS -l vmem=8gb,mem=8gb
# Standard error & standard output are merged in example.out
#PBS -j oe
#PBS -o submitjob.out
# Sends a mail when the job ends
#PBS -m e

# Good idea to only load module needed by the job
module purge
# Load python 3.9 module and GCC > 4.8
module load python/meso-3.9
module load gnu/4.9.3

# activate the virtual environment
source /home/cmarmo/boomenv/bin/activate

# do something
python --version
python -c "import sklearn; sklearn.show_versions()"

# deactivate the virtual environment
deactivate

# unload python 3.9
module purge

