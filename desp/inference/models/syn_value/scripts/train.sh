export BATCH_SIZE=4096
export NUM_NODES=1
export NUM_GPU=1
export DROPOUT=0.3
export LR=0.003569581152828107
export DATA_NAME="USPTO"
export NUM_CORES=32

python ./train.py \
  --backend=nccl \
  --model_name="syn_dist" \
  --data_name="$DATA_NAME" \
  --log_file="syn_dist_$DATA_NAME" \
  --processed_data_path=./data/ \
  --model_type='dist' \
  --model_path=./checkpoints/ \
  --input_type="concat" \
  --max_label=9 \
  --fp_size=512 \
  --dropout="$DROPOUT" \
  --seed=42 \
  --epochs=150 \
  --hidden_activation="sigmoid" \
  --hidden_sizes=1024,1024,1024 \
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