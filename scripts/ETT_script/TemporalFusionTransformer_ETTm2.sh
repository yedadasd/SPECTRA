model_name=TemporalFusionTransformer
seq_len=168

for pred_len in 48 96 192 336
# for pred_len in 336 720
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
  --label_len 12 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
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