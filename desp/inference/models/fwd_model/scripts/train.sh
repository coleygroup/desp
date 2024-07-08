export BATCH_SIZE=4096
export NUM_NODES=1
export NUM_GPU=1
export DROPOUT=0.3
export LR=0.001551441107365456
export DATA_NAME="USPTO"
export NUM_CORES=32

python ./fwd_trainer.py \
  --backend=nccl \
  --model_name="fwd_template_relevance" \
  --data_name="$DATA_NAME" \
  --log_file="template_relevance_train_$DATA_NAME" \
  --processed_data_path=./data/ \
  --model_path=./checkpoints/ \
  --model_type="bb" \
  --pos_weight=1 \
  --dropout="$DROPOUT" \
  --seed=42 \
  --epochs=150 \
  --hidden_sizes=2048,2048,2048 \
  --hidden_activation="relu" \
  --learning_rate="$LR" \
  --batch_size="$BATCH_SIZE" \
  --train_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --val_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --test_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --num_cores="$NUM_CORES" \
  --lr_scheduler_factor 0.3 \
  --lr_scheduler_patience 1 \
  --early_stop True \
  --early_stop_patience 2 \
  --early_stop_min_delta 0