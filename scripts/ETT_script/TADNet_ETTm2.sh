model_name=TADNet

seq_len=168

for pred_len in 12 24 36 72 120 168
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
  --enc_in 7 \
  --des 'TADNet' \
  --d_model 160 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --n_heads 4 \
  --dropout 0.2 \
  --train_epochs 10 \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --step 1 \
  --itr 1 \

done
