#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

nohup bash ./scripts/long_term_forecast_hyperS/ETT_script/PatchTST_ETTh1.sh > ./log_20240909/ETT_script_PatchTST_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/Crossformer_ETTh1.sh > ./log_20240909/ETT_script_Crossformer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/Autoformer_ETTh1.sh > ./log_20240909/ETT_script_Autoformer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/Nonstationary_Transformer_ETTh1.sh > ./log_20240909/ETT_script_Nonstationary_Transformer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/TimeMixer_ETTh1.sh > ./log_20240909/ETT_script_TimeMixer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/PatchTST.sh > ./log_20240909/ECL_script_PatchTST.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/Crossformer.sh > ./log_20240909/ECL_script_Crossformer.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/Autoformer.sh > ./log_20240909/ECL_script_Autoformer.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/Nonstationary_Transformer.sh > ./log_20240909/ECL_script_Nonstationary_Transformer.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/TimeMixer.sh > ./log_20240909/ECL_script_TimeMixer.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/PatchTST.sh > ./log_20240909/Weather_script_PatchTST.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/Crossformer.sh > ./log_20240909/Weather_script_Crossformer.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/Autoformer.sh > ./log_20240909/Weather_script_Autoformer.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/Nonstationary_Transformer.sh > ./log_20240909/Weather_script_Nonstationary_Transformer.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/TimeMixer.sh > ./log_20240909/Weather_script_TimeMixer.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/iTransformer_ETTh1.sh > ./log_20240909/ETT_script_iTransformer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ETT_script/Transformer_ETTh1.sh > ./log_20240909/ETT_script_Transformer_ETTh1.out &
bash ./scripts/long_term_forecast_hyperS/ECL_script/iTransformer.sh > ./log_20240909/ECL_script_iTransformer &
bash ./scripts/long_term_forecast_hyperS/ECL_script/Transformer.sh > ./log_20240909/ECL_script_Transformer &
bash ./scripts/long_term_forecast_hyperS/Weather_script/iTransformer.sh > ./log_20240909/Weather_script_iTransformer.out &
bash ./scripts/long_term_forecast_hyperS/Weather_script/Transformer.sh > ./log_20240909/Weather_script_Transformer &

conda deactivate
