#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

bash ./scripts/long_term_forecast_hyperS/ETT_script/PatchTST_ETTh1.sh > ./log_20240909/ETTh1_script_PatchTST_ETTh1
bash ./scripts/long_term_forecast_hyperS/ETT_script/Crossformer_ETTh1.sh > ./log_20240909/ETTh1_script_Crossformer_ETTh1
bash ./scripts/long_term_forecast_hyperS/ETT_script/Autoformer_ETTh1.sh > ./log_20240909/ETTh1_script_Autoformer_ETTh1 
bash ./scripts/long_term_forecast_hyperS/ETT_script/Nonstationary_Transformer_ETTh1.sh > ./log_20240909/ETTh1_script_Nonstationary_Transformer_ETTh1
bash ./scripts/long_term_forecast_hyperS/ETT_script/TimeMixer_ETTh1.sh > ./log_20240909/ETTh1_script_TimeMixer_ETTh1
bash ./scripts/long_term_forecast_hyperS/ECL_script/PatchTST.sh > ./log_20240909/ECL_script_PatchTST
bash ./scripts/long_term_forecast_hyperS/ECL_script/Crossformer.sh > ./log_20240909/ECL_script_Crossformer
bash ./scripts/long_term_forecast_hyperS/ECL_script/Autoformer.sh > ./log_20240909/ECL_script_Autoformer
bash ./scripts/long_term_forecast_hyperS/ECL_script/Nonstationary_Transformer.sh > ./log_20240909/ECL_script_Nonstationary_Transformer
bash ./scripts/long_term_forecast_hyperS/ECL_script/TimeMixer.sh > ./log_20240909/ECL_script_TimeMixer 
bash ./scripts/long_term_forecast_hyperS/Weather_script/PatchTST.sh > ./log_20240909/weather_script_PatchTST 
bash ./scripts/long_term_forecast_hyperS/Weather_script/Crossformer.sh > ./log_20240909/weather_script_Crossformer 
bash ./scripts/long_term_forecast_hyperS/Weather_script/Autoformer.sh > ./log_20240909/weather_script_Autoformer 
bash ./scripts/long_term_forecast_hyperS/Weather_script/Nonstationary_Transformer.sh > ./log_20240909/weather_script_Nonstationary_Transformer 
bash ./scripts/long_term_forecast_hyperS/Weather_script/TimeMixer.sh > ./log_20240909/weather_script_TimeMixer 

conda deactivate
