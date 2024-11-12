#!/bin/bash
source /home/jxu/miniconda3/etc/profile.d/conda.sh
conda activate time_hyperS_0909

bash ./scripts/long_term_forecast_hyperS/ECL_script/PatchTST.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/Crossformer.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/Autoformer.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/Nonstationary_Transformer.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/iTransformer.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/Transformer.sh
bash ./scripts/long_term_forecast_hyperS/ECL_script/TimeMixer.sh

conda deactivate
