model_name=SPECTRA_I
seq_len=168

for zone in 1 2 3
do

for pred_len in 12 24 36
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Solar/ \
  --data_path Solarz${zone}_OT.csv \
  --model_id GEFCom_Solarz${zone}_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 13 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 13 \
  --des '' \
  --d_model 128 \
  --batch_size 16 \
  --use_norm 1 \
  --learning_rate 0.0002 \
  --loss_type quantileLoss \
  --loss_alpha 0.8 \
  --dropout 0.2 \
  --wavelet_type haar \
  --norm_method RevIN \
  --cross_attn_heads 8 \
  --step 1 \
  --cutoff_freq 0.25 \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

for pred_len in 72
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/GEFCom2014/Solar/ \
  --data_path Solarz${zone}_OT.csv \
  --model_id GEFCom_Solarz${zone}_$seq_len'_'$pred_len \
  --model $model_name \
  --data GEFCom \
  --target OT \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 13 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 13 \
  --des '' \
  --d_model 128 \
  --batch_size 16 \
  --use_norm 1 \
  --learning_rate 0.0003 \
  --loss_type quantileLoss \
  --cross_attn_heads 2 \
  --loss_alpha 0.8 \
  --dropout 0.2 \
  --wavelet_type haar \
  --norm_method RevIN \
  --step 1 \
  --cutoff_freq 0.35 \
  --itr 1 \
  --quantiles "[0.1, 0.5, 0.9]"

done

done
