seq_len=168
model_name=DLinear

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
  --pred_len $pred_len \
  --enc_in 59 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --step 1

done