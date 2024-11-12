model_name=Autoformer
root_path='/home/jxu/Documents/time_hyperS_project/Time-Series-Library_HyperS_20240909/dataset/electricity/'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --extra_tag 'BasicVariantGenerator'

