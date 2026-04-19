#!/bin/bash
model_name=DeepAR
seq_len=168
d_model=128

for pred_len in 12 24 36 72 120 168
do

python -u run.py \
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
  --e_layers 2 \
  --d_model $d_model \
  --enc_in 59 \
  --dec_in 59 \
  --c_out 59 \
  --des 'Exp' \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1 \
  --batch_size 32 \
  --step 12 \
  --learning_rate 0.001

done