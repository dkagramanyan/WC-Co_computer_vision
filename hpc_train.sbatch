#!/bin/bash
#SBATCH --job-name=autoencoder_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task 6

module load Python/Anaconda_v10.2019

# Активация окружения Anaconda
# Перед активацией окружения рекомендуется деактивировать текущее
source deactivate
source activate tf2

echo "Working on node `hostname`"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
srun python nn_train.py