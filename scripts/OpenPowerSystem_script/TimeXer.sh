#!/bin/bash
model_name=TimeXer
seq_len=168

for pred_len in 12 24 36 72 120 168
do

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/OpenPowerSystem/load/ \
  --data_path load.csv \
  --model_id OpenPower_load_$seq_len'_'$pred_len \
  --model $model_name \
  --data OpenPower \
  --target NO_2_load_actual_entsoe_transparency \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --factor 3 \
  --enc_in 59 \
  --dec_in 59 \
  --c_out 59 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --des 'exp' \
  --loss_type quantileLoss \
  --step 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1

done