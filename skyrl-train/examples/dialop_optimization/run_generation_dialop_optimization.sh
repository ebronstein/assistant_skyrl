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

ASYNC_ENGINE=false
INFERENCE_BACKEND="vllm"
ASSISTANT_MODEL="microsoft/Phi-4-mini-instruct"
USER_MODEL="microsoft/Phi-4-mini-instruct"
GPU_MEMORY_UTILIZATION=0.4

uv run --active --no-sync --extra $INFERENCE_BACKEND --env-file .env \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.skyrl_gym.dialop_optimization.model=$USER_MODEL \
  trainer.export_path="$PROJECT_DIR/exports/dialop_optimization" \
  trainer.policy.model.path=$ASSISTANT_MODEL \
  trainer.logger="$LOGGER" \
  trainer.max_prompt_length=16384 \
  generator.max_num_batched_tokens=16384 \
  +generator.engine_init_kwargs.max_model_len=16384 \
  generator.max_turns=2 \
  generator.async_engine=$ASYNC_ENGINE \
  generator.batched=$([[ "$ASYNC_ENGINE" == "false" ]] && echo "true" || echo "false") \
  generator.backend=$INFERENCE_BACKEND \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.top_p=0.9 \
  environment.env_class=dialop_optimization \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  +generator.engine_init_kwargs.generation_config=vllm \
  "$@"
