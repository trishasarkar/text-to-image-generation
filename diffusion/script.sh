#!/bin/bash
#SBATCH --job-name=latentdiffusion
#SBATCH --output=vllm_v4.out
#SBATCH --error=vllm_v4.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48Gb
#SBATCH --time=0-24:00:00

source /home/amysore2/.bashrc
conda activate latentdiffusion

# long context cmd
python main.py --save_folder run_3 --time_steps 1000 --save_and_sample_every 10000 --train_steps 200000 --image_size 64 --batch_size 1 --learning_rate 1e-5 --unet_dim 16 --load_path "run_2/model_2.pt"