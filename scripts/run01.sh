#!/bin/bash

## Old workload manager params
#$ -cwd
#$ -o ../logs/
#$ -e ../logs/

## Slurm params
#SBATCH -p long
#SBATCH -o ../logs/%j.o
#SBATCH -e ../logs/%j.e
#SBATCH --mail-user=charlie.clark@balliol.ox.ac.uk
#SBATCH --mail-type=ALL

source ~/.bashrc

module purge

module add Python/3.8.2-GCCcore-9.3.0
module add SciPy-bundle/2020.03-foss-2020a-Python-3.8.2
module add scikit-learn/0.23.1-foss-2020a-Python-3.8.2
module add matplotlib/3.2.1-foss-2020a-Python-3.8.2
module add Seaborn/0.11.2-foss-2021a

conda run --no-capture-output --name rf python -u _01_annotate.py $1 $2 $3
