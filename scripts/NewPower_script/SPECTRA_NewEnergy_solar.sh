#!/bin/bash
# SPECTRA_I on NewEnergy Solar (stations 6-10, 15-min resolution)
# Solar has strong diurnal pattern — larger d_state & db4 wavelet for periodicity
# v2 data: 25 features (24 NWP + 1 OT) → enc_in=25, c_out=25

model_name=SPECTRA_I
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
  --model_id NewEnergy_Solar${station}_${seq_len}_${pred_len} \
  --model $model_name \
  --data NewEnergy \
  --target OT \
  --features M \
  --freq 15min \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in $enc_in \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out $c_out \
  --d_model 256 \
  --batch_size 32 \
  --use_norm 1 \
  --dropout 0.15 \
  --learning_rate 0.0005 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
  --cutoff_freq 0.45 \
  --step 1 \
  --norm_method NS \
  --wavelet_type db4 \
  --cross_attn_heads 4 \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

# ---- long horizon: 1d, 2d ----
for pred_len in 96 192
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NewEnergy/ \
  --data_path solar${station}${data_ver}.csv \
  --model_id NewEnergy_Solar${station}_${seq_len}_${pred_len} \
  --model $model_name \
  --data NewEnergy \
  --target OT \
  --features M \
  --freq 15min \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in $enc_in \
  --expand 2 \
  --d_state 32 \
  --d_conv 4 \
  --c_out $c_out \
  --d_model 128 \

done
