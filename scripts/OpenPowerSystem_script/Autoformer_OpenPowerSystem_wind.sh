#!/bin/bash
model_name=Autoformer
seq_len=168

for pred_len in 12 24 36 72 120 168
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/OpenPowerSystem/wind/ \
  --data_path wind.csv \
  --model_id OpenPower_wind_$seq_len'_'$pred_len \
  --model $model_name \
  --data OpenPower \
  --target LT_wind_onshore_generation_actual \
  --features M \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 57 \
  --dec_in 57 \
  --c_out 57 \
  --des 'Exp' \
  --loss_type quantileLoss \
  --step 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1

done
