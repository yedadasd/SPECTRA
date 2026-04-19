model_name=TemporalFusionTransformer
seq_len=168

for pred_len in 36 168
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/OpenPowerSystem/solar/ \
  --data_path solar.csv \
  --model_id OpenPower_solar_$seq_len'_'$pred_len \
  --model $model_name \
  --data OpenPower \
  --target LT_solar_generation_actual \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --des 'Exp' \
  --batch_size 8 \
  --loss_type quantileLoss \
  --step 12 \
  --quantiles "[0.1, 0.5, 0.9]" \
  --d_ff 512 \
  --d_model 512 \
  --dropout 0.3 \
  --itr 1

done
