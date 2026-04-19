model_name=TiDE
seq_len=168

for pred_len in 12 24 36 72 120 168
# for pred_len in 336 720
do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 2 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 8 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.3 \
    --learning_rate 0.1 \
    --patience 5 \
    --des 'Exp' \
    --loss_type quantileLoss \
    --step 1 \
    --quantiles "[0.1, 0.5, 0.9]" \
    --itr 1 \

done