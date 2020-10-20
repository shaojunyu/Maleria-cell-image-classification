#!/bin/sh

#SBATCH -J Cell_Img
#SBATCH -p overflow
#SBATCH -G 1
#SBATCH -t 3:0:0
#SBATCH -o ./malaria_%A.out
#SBATCH -e ./malaria_%A.err
#SBATCH --mem 16G

hostname
scl enable rh-python36 'python BMI500_malaria_cells.py'

