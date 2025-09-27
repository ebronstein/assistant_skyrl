# Generation only for for Qwen2.5-1.5B-Instruct on DialOp Optimization.

# If you need to use a shorter max sequence length for the assistant model (e.g., due
# to memory constraints), you can initialize the vLLM engine with the `max_model_len`
# argument by setting the `+generator.engine_init_kwargs.max_model_len` flag.
# Make sure `trainer.max_prompt_length` and `generator.max_num_batched_tokens` are
# no greater than `max_model_len`.

HOME_DIR="/nas/ucb/$USER"
PROJECT_DIR="$HOME_DIR/assistant_skyrl"
DATA_DIR="$PROJECT_DIR/data/dialop_optimization"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

ASSISTANT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
USER_MODEL="meta-llama/Llama-3.1-8B-Instruct"

ASYNC_ENGINE=true
BATCHED=$([[ "$ASYNC_ENGINE" == "false" ]] && echo "true" || echo "false")
INFERENCE_BACKEND="vllm"
GPU_MEMORY_UTILIZATION=0.4

MAX_PROMPT_LENGTH=16384
MAX_NUM_BATCHED_TOKENS=8192
# Ensure max_num_batched_tokens is not greater than max_prompt_length
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_LENGTH < MAX_NUM_BATCHED_TOKENS ? MAX_PROMPT_LENGTH : MAX_NUM_BATCHED_TOKENS ))

# NOTE (ebronstein): trainer.placement.colocate_all=false is needed to fix a bug where
# the model weights are corrupted when calling `wake_up()` on the inference engine.
uv run --active --no-sync --extra $INFERENCE_BACKEND --env-file .env \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.export_path="$PROJECT_DIR/exports/dialop_optimization" \
  trainer.policy.model.path=$ASSISTANT_MODEL \
  trainer.logger="$LOGGER" \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.placement.colocate_all=false \
  generator.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  +generator.engine_init_kwargs.max_model_len=$MAX_PROMPT_LENGTH \
  generator.async_engine=$ASYNC_ENGINE \
  generator.batched=$BATCHED \
  generator.backend=$INFERENCE_BACKEND \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  environment.skyrl_gym.dialop_optimization.user_model=$USER_MODEL \
  environment.env_class=dialop_optimization \
  generator.max_turns=10 \
  generator.eval_sampling_params.temperature=0.7 \
  "$@"
