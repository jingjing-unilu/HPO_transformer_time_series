#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

bash ./scripts/long_term_forecast_hyperS/ETT_script/Mamba_ETTh1.sh > ./log_20240909/ETT_script_Mamba_ETTh1
bash ./scripts/long_term_forecast_hyperS/ECL_script/Mamba.sh > ./log_20240909/ECL_script_Mamba
bash ./scripts/long_term_forecast_hyperS/Weather_script/Mamba.sh > ./log_20240909/Weather_script_Mamba 

conda deactivate
