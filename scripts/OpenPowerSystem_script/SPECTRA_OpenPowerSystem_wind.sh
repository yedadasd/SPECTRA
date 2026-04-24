model_name=SPECTRA_I
seq_len=168

for pred_len in 12 24 36 72 120 168
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/OpenPowerSystem/wind/ \
  --data_path wind.csv \
  --model_id OpenPower_wind_$seq_len'_'$pred_len \
  --model $model_name \
  --data OpenPower \
  --target LT_wind_onshore_generation_actual \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 57 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 57 \
  --des '' \
  --d_model 256 \
  --batch_size 32 \
  --clip \
  --use_norm 1 \
  --learning_rate 0.0001 \
  --loss_type quantileLoss \
  --dropout 0.3 \
  --cutoff_freq 0.125 \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done
