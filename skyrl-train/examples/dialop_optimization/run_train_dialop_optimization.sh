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
GPU_MEMORY_UTILIZATION=0.2

MAX_PROMPT_LENGTH=8192
MAX_NUM_BATCHED_TOKENS=8192
# Ensure max_num_batched_tokens is not greater than max_prompt_length
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_LENGTH < MAX_NUM_BATCHED_TOKENS ? MAX_PROMPT_LENGTH : MAX_NUM_BATCHED_TOKENS ))

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}  # original: 1024
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}  # original: 1024
POLICY_MINI_BATCH_SIZE=${POLICY_MINI_BATCH_SIZE:-32}  # original: 256
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}  # original: 5

uv run --active --no-sync --extra $INFERENCE_BACKEND --env-file .env -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$ASSISTANT_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=$EVAL_BATCH_SIZE \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  +generator.engine_init_kwargs.max_model_len=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=1024 \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=$ASYNC_ENGINE \
  generator.batched=$BATCHED \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  trainer.logger="$LOGGER" \
  trainer.project_name="dialop_optimization" \
  trainer.run_name="dialop_optimization_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$PROJECT_DIR/ckpts/dialop_optimization_Qwen2.5-1.5B-Instruct_ckpt" \
  environment.env_class=dialop_optimization \
  environment.skyrl_gym.dialop_optimization.user_model=$USER_MODEL \
  $@
