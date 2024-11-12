#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/PatchTST_ETTh1.sh > ./log_20240916/ETTh1_script_PatchTST_ETTh1
bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/Crossformer_ETTh1.sh > ./log_20240916/ETTh1_script_Crossformer_ETTh1
bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/Autoformer_ETTh1.sh > ./log_20240916/ETTh1_script_Autoformer_ETTh1 
bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/Nonstationary_Transformer_ETTh1.sh > ./log_20240916/ETTh1_script_Nonstationary_Transformer_ETTh1
bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/TimeMixer_ETTh1.sh > ./log_20240916/ETTh1_script_TimeMixer_ETTh1
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/PatchTST.sh > ./log_20240916/ECL_script_PatchTST
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/Crossformer.sh > ./log_20240916/ECL_script_Crossformer
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/Autoformer.sh > ./log_20240916/ECL_script_Autoformer
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/Nonstationary_Transformer.sh > ./log_20240916/ECL_script_Nonstationary_Transformer
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/TimeMixer.sh > ./log_20240916/ECL_script_TimeMixer 
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/PatchTST.sh > ./log_20240916/weather_script_PatchTST 
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/Crossformer.sh > ./log_20240916/weather_script_Crossformer 
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/Autoformer.sh > ./log_20240916/weather_script_Autoformer 
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/Nonstationary_Transformer.sh > ./log_20240916/weather_script_Nonstationary_Transformer 
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/TimeMixer.sh > ./log_20240916/weather_script_TimeMixer
bash ./scripts/long_term_forecast_hyperS_grid_search/ETT_script/Mamba_ETTh1.sh > ./log_20240909/ETTh1_script_Mamba_ETTh1
bash ./scripts/long_term_forecast_hyperS_grid_search/ECL_script/Mamba.sh > ./log_20240909/ECL_script_Mamba
bash ./scripts/long_term_forecast_hyperS_grid_search/Weather_script/Mamba.sh > ./log_20240909/weather_script_Mamba 


conda deactivate
