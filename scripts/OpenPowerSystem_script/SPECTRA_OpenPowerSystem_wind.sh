model_name=SPECTRA_I
seq_len=168

for pred_len in 12 24 36 
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
  --e_layers 1 \
  --enc_in 57 \
  --expand 1 \
  --d_state 8 \
  --d_conv 2 \
  --c_out 57 \
  --des '' \
  --d_model 128 \
  --batch_size 32 \
  --clip \
  --use_norm 1 \
  --learning_rate 0.0003 \
  --loss_type quantileLoss \
  --dropout 0 \
  --cutoff_freq 0.25 \
  --step 1 \
  --norm_method RevIN \
  --wavelet_type haar \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --cross_attn_heads 4 

done

for pred_len in 72
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
  --e_layers 3 \
  --enc_in 57 \
  --expand 2 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 57 \
  --des '' \
  --d_model 256 \
  --batch_size 64 \
  --clip \
  --use_norm 1 \
  --learning_rate 0.0002 \
  --loss_type quantileLoss \
  --dropout 0 \
  --cutoff_freq 0.1 \
  --step 1 \
  --norm_method RevIN \
  --wavelet_type db2 \
  --itr 1 \
  --cross_attn_heads 2 \
  --quantiles "[0.1, 0.5, 0.9]"

done