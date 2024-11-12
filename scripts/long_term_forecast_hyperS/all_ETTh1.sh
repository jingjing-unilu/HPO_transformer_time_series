#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

bash ./scripts/long_term_forecast_hyperS/ETT_script/PatchTST_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/Crossformer_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/Autoformer_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/Nonstationary_Transformer_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/iTransformer_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/Transformer_ETTh1.sh
bash ./scripts/long_term_forecast_hyperS/ETT_script/TimeMixer_ETTh1.sh

conda deactivate
