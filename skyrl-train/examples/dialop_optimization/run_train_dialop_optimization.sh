HOME_DIR="/nas/ucb/$USER"
PROJECT_DIR="$HOME_DIR/assistant_skyrl"
DATA_DIR="$PROJECT_DIR/data/dialop_optimization"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
POLICY_MINI_BATCH_SIZE=4

uv run --active --no-sync --extra $INFERENCE_BACKEND --env-file .env -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=$TRAIN_BATCH_SIZE \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=$POLICY_MINI_BATCH_SIZE \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="dialop_optimization" \
  trainer.run_name="dialop_optimization_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$PROJECT_DIR/ckpts/dialop_optimization_1.5B_ckpt" \
  environment.env_class=dialop_optimization \
  environment.skyrl_gym.dialop_optimization.model=$MODEL \
  environment.skyrl_gym.dialop_optimization.openai_client.api_key="EMPTY" \
  environment.skyrl_gym.dialop_optimization.openai_client.api_base="http://localhost:8000/v1" \
  $@
