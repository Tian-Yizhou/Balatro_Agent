#!/bin/bash
# Full PPO training run on Balatro easy mode.
#
# Use this for real experiments. run_exp.sh is the 200-iter smoke test;
# this script trains longer with tuned hyperparameters based on what the
# pilot run showed (reward plateau around iter 50, win_rate still climbing
# at iter 200 -> needs more samples to converge).

set -euo pipefail

# ---- activate conda environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate balatro-agent

# ---- wandb mode ----
# "online"  -> push live to wandb.ai (requires `wandb login` once)
# "offline" -> cache locally, sync later with `wandb sync`
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---- run tag (timestamp) ----
# Each run gets its own checkpoint dir and log file so reruns don't overwrite.
DIFFICULTY="${DIFFICULTY:-easy}"
RUN_TAG="${DIFFICULTY}_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/ppo_${RUN_TAG}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/train_${RUN_TAG}.log"
mkdir -p "$LOG_DIR"

# ---- train ----
python -m agent.rllib.train \
    --difficulty "$DIFFICULTY" \
    --num-iterations 1000 \
    --num-env-runners 8 \
    --num-envs-per-env-runner 1 \
    --num-gpus-per-learner 1 \
    --train-batch-size 8000 \
    --sgd-minibatch-size 512 \
    --num-epochs 10 \
    --lr 3e-4 \
    --gamma 0.99 \
    --lambda-gae 0.95 \
    --clip-param 0.2 \
    --entropy-coeff 0.01 \
    --vf-loss-coeff 0.5 \
    --fcnet-hiddens 256 256 \
    --fcnet-activation relu \
    --checkpoint-freq 25 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    # --restore-from "checkpoints/ppo_easy_20260528_202936/latest" \
    --wandb-project balatro-agent \
    --wandb-run-name "ppo_${RUN_TAG}" \
    2>&1 | tee "$LOG_FILE"

echo
echo "Run complete."
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Log file:    $LOG_FILE"
echo "  Metrics:     $CHECKPOINT_DIR/metrics.json"
