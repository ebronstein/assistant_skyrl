# Generate data for DialOp Optimization.

DATA_DIR=/nas/ucb/$USER/assistant_skyrl/data/dialop_optimization
NUM_ASSIGNMENTS=8
P_CELL_OBSERVED=0.4
TRAIN_SIZE=20
TEST_SIZE=5

uv run --active --no-sync -m skyrl_gym.envs.dialop_optimization.dataset \
    --output_dir $DATA_DIR \
    --num_assignments $NUM_ASSIGNMENTS \
    --p_cell_observed $P_CELL_OBSERVED \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE
