model_name=SPECTRA_I
seq_len=168

for pred_len in 12 24 36
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Price/ \
  --data_path Price_OT.csv \
  --model_id GEFCom_Price_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 3 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 3 \
  --des '' \
  --d_model 256 \
  --batch_size 32 \
  --use_norm 1 \
  --learning_rate 0.0002 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
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
  --root_path ./dataset/GEFCom2014/Price/ \
  --data_path Price_OT.csv \
  --model_id GEFCom_Price_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 3 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 3 \
  --des '' \
  --d_model 256 \
  --batch_size 32 \
  --use_norm 1 \
  --learning_rate 0.0001 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done