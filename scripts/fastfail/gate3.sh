#!/bin/bash
# Gate 3: official Flow-GRPO, smallest documented config, one Max-Q (GPU 0). Eval/checkpoints pushed out of reach.
# FASTFAIL_ROOT: dir holding venv/, src/flow_grpo, hf/ (default: current dir)
R=${FASTFAIL_ROOT:-$PWD}; RUN=$R/runs/gate3-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $RUN
cd $R/src/flow_grpo
export HF_HOME=$R/hf HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR=$RUN CUDA_VISIBLE_DEVICES=0 PADDLE_HOME=$R/cache/paddle XDG_CACHE_HOME=$R/cache HOME_PADDLEOCR=$R/cache/paddleocr
{ echo "run $RUN"; date -u; git log -1 --format='flow_grpo %H'; nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader; $R/venv/bin/python -c "import torch,diffusers,transformers,accelerate,peft;print('torch',torch.__version__,'cuda',torch.version.cuda,'diffusers',diffusers.__version__,'tf',transformers.__version__,'accelerate',accelerate.__version__,'peft',peft.__version__)"; sed -n '/def general_ocr_sd3_1gpu/,/return config/p' config/grpo.py; } > $RUN/versions.txt 2>&1
cp $R/freeze.txt $RUN/
( while sleep 15; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader >> $RUN/gpu.csv; done ) & MON=$!
$R/venv/bin/accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 \
  scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu \
  --config.eval_freq=100000 --config.save_freq=100000 --config.num_epochs=20 \
  --config.logdir=$RUN/logs --config.save_dir=$RUN/save > $RUN/train.log 2>&1
echo "exit $?" >> $RUN/train.log; kill $MON
