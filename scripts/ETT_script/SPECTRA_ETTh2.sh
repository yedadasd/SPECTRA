model_name=SPECTRA_I

seq_len=168

for pred_len in 12 24 36 72 120 168
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
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 256 \
  --use_norm 1 \
  --des '' \
  --batch_size 32 \
  --loss_type quantileLoss \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  # --norm_method RevIN \
done