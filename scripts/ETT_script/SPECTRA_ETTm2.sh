model_name=SPECTRA_I

seq_len=168

for pred_len in 48 96 192 336
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 128 \
  --use_norm 1 \
  --des '' \
  --batch_size 32 \
  --loss_type quantileLoss \
  --step 1 \
  --norm_method RevIN \
  --itr 1 \
  --learning_rate 0.0005 \
  --dropout 0.15 \
  --cutoff_freq 0.20 \

done