#!/bin/bash
model_name=TiDE
seq_len=168

for pred_len in 12 24 36 72 120 168
do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/OpenPowerSystem/price/ \
    --data_path price.csv \
    --model_id OpenPower_price_$seq_len'_'$pred_len \
    --model $model_name \
    --data OpenPower \
    --target IT_NORD_CH_price_day_ahead \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 2 \
    --factor 3 \
    --enc_in 31 \
    --dec_in 31 \
    --c_out 32 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.3 \
    --learning_rate 0.1 \
    --patience 5 \
    --des 'Exp' \
    --loss_type quantileLoss \
    --step 1 \
    --quantiles "[0.1, 0.5, 0.9]" \
    --itr 1

done
