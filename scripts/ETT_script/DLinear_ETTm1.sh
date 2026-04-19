seq_len=168
model_name=DLinear

for pred_len in 48 96 192 336
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --step 1

done