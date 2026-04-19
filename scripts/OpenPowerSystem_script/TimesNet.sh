model_name=TimesNet
seq_len=168

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
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 59 \
  --dec_in 59 \
  --c_out 59 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --loss_type quantileLoss \
  --step 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --itr 1

done