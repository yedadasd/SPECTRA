model_name=SPECTRA_I
seq_len=168

for zone in 1 5 10
do

for pred_len in 12 24 36
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Wind/ \
  --data_path Windz${zone}_OT.csv \
  --model_id GEFCom_Windz${zone}_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 5 \
  --expand 1 \
  --d_state 8 \
  --d_conv 2 \
  --c_out 5 \
  --des '' \
  --d_model 512 \
  --batch_size 32 \
  --use_norm 1 \
  --dropout 0.15 \
  --learning_rate 0.0056 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
  --cutoff_freq 0.458 \
  --step 1 \
  --norm_method NS \
  --wavelet_type haar \
  --cross_attn_heads 2 \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

for pred_len in 72
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Wind/ \
  --data_path Windz${zone}_OT.csv \
  --model_id GEFCom_Windz${zone}_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 5 \
  --expand 2 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 5 \
  --des '' \
  --d_model 128 \
  --batch_size 16 \
  --use_norm 1 \
  --dropout 0.3 \
  --learning_rate 0.005 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
  --cutoff_freq 0.35 \
  --step 1 \
  --norm_method NS \
  --wavelet_type db4 \
  --cross_attn_heads 8 \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

done
