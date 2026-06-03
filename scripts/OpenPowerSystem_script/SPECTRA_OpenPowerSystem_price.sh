model_name=SPECTRA_I
seq_len=168

for pred_len in 12 24 36 
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
  --enc_in 31 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 31 \
  --des '' \
  --d_model 256 \
  --batch_size 16 \
  --clip \
  --use_norm 1 \
  --learning_rate 0.0001 \
  --loss_type quantileLoss \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

for pred_len in 72 120 168
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
  --e_layers 3 \
  --enc_in 31 \
  --expand 1 \
  --d_state 32 \
  --d_conv 2 \
  --c_out 31 \
  --des '' \
  --d_model 256 \
  --batch_size 32 \
  --clip \
  --use_norm 1 \
  --learning_rate 0.0001 \
  --loss_type quantileLoss \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done
