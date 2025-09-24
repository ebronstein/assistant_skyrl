# Generation only for for Qwen2.5-1.5B-Instruct on DialOp Optimization.
HOME_DIR="/nas/ucb/$USER"
PROJECT_DIR="$HOME_DIR/assistant_skyrl"
DATA_DIR="$PROJECT_DIR/data/dialop_optimization"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

uv run --active --no-sync --extra $INFERENCE_BACKEND \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path=$MODEL \
  trainer.logger="$LOGGER" \
  generator.backend=$INFERENCE_BACKEND \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=dialop_optimization \
  "$@"
