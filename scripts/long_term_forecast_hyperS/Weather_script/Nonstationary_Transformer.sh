model_name=Nonstationary_Transformer
root_path='/home/jxu/Documents/time_hyperS_project/Time-Series-Library_HyperS_20240909/dataset/weather/'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --extra_tag 'OptunaSearch'
  

