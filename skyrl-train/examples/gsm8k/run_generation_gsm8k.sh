# set -x

# Generation only for for Qwen2.5-0.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_generation_gsm8k.sh

HOME_DIR="/nas/ucb/$USER"
PROJECT_DIR="$HOME_DIR/assistant_skyrl"
DATA_DIR="$PROJECT_DIR/data/gsm8k"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"  # or "sglang"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
GPU_MEMORY_UTILIZATION=0.4

uv run --active --no-sync --extra $INFERENCE_BACKEND --env-file .env \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="$MODEL" \
  trainer.logger="$LOGGER" \
  generator.backend=$INFERENCE_BACKEND \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=gsm8k \
  "$@"
