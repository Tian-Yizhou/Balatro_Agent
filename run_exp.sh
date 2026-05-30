# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate balatro-agent

WANDB_MODE="offline"

# run experiment
python -m agent.rllib.train \
      --difficulty easy \
      --num-env-runners 8 \
      --num-gpus-per-learner 1 \
      --num-iterations 200 \
      --checkpoint-dir checkpoints/balatro_ppo_easy 2>&1 | tee "train_$(date +%Y%m%d_%H%M%S).log"
