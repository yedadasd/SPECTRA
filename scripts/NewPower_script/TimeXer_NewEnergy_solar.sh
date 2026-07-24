#!/bin/bash
# TimeXer on NewEnergy Solar (stations 6-10, 15-min resolution)
# v2 data: 25 features → enc_in=25, c_out=25

model_name=TimeXer
seq_len=96
data_ver="_v2"       # change to "" for v1
enc_in=25             # change to 10 for v1
c_out=25              # change to 10 for v1

for station in 7 8
do

for pred_len in 96
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NewEnergy/ \
  --data_path solar${station}${data_ver}.csv \
  --model_id NewEnergy_Solar${station}_TimeXer_${seq_len}_${pred_len} \
  --model $model_name \
  --data NewEnergy \
  --target OT \
  --features M \
  --freq 15min \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $c_out \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --des 'exp' \
  --loss_type quantileLoss \
  --step 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1

done

done
