model_name=TADNet
seq_len=168

for pred_len in 12 24 36 72 120 168
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Load/ \
  --data_path Load_OT.csv \
  --model_id GEFCom_Load_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --enc_in 26 \
  --des 'TADNet' \
  --d_model 160 \
  --batch_size 32 \
  --n_heads 4 \
  --dropout 0.2 \
  --train_epochs 10 \
  --loss_type quantileLoss \
  --quantiles "[0.1, 0.5, 0.9]" \
  --step 1 \
  --itr 1

done
