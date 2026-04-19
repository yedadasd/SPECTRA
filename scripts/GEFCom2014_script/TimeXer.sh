#!/bin/bash
model_name=TimeXer
seq_len=168

for pred_len in 12 24 36 72 120 168
do

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Load/ \
  --data_path Load_OT.csv \
  --model_id GEFCom_Load_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --factor 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --des 'exp' \
  --loss_type quantileLoss \
  --step 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1

done