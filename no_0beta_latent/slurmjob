#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gen_dm_4x4 
# Queue (Partition):
#SBATCH --partition=compIntel
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=frohnert@mail.lorentz.leidenuniv.nl
#
# Wall clock limit:
#SBATCH --time=15:00:00
#Load some modules
#module load Python/3.9.5-GCCcore-10.3.0
module load QuantumMiniconda3/4.7.10
#module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
source /marisdata/frohnert/cluster/gen_dm/venv/bin/activate

size=1 
beta=0.0
noise="False"
for n_lat in 2
do
   srun python /home/frohnert/cluster/vae_4x4_mult.py $n_lat $size $beta $noise
done
