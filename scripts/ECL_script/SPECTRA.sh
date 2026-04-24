model_name=SPECTRA_I

seq_len=168

for pred_len in 12 24 36 72 120 168
# for pred_len in 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 321 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 321 \
  --des '' \
  --d_model 512 \
  --batch_size 32 \
  --use_norm 1 \
  --learning_rate 0.0005 \
  --loss_type quantileLoss \
  --step 1 \
  --norm_method NS \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  # --norm_method RevIN \
done
