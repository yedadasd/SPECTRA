#!/bin/bash
model_name=DeepAR
seq_len=168
d_model=128

for pred_len in 12 24 36 72 120 168
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/OpenPowerSystem/solar/ \
  --data_path solar.csv \
  --model_id OpenPower_solar_$seq_len'_'$pred_len \
  --model $model_name \
  --data OpenPower \
  --target LT_solar_generation_actual \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_model $d_model \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --des 'Exp' \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1 \
  --batch_size 32 \
  --step 1 \
  --learning_rate 0.001

done
