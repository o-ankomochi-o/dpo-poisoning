#!/bin/bash
#$ -l rt_G.large=1
#$ -j y
#$ -N dpo_training
#$ -o logs/
#$ -cwd

# 環境設定
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 hpcx/2.12 gcc/13.2.0 nccl/2.14/2.14.3-1 
source .venv/bin/activate

# 基本的な環境設定
export PYTHONPATH="/home/acg16509aq/ogawa/rlhf-poisoning:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOGLEVEL=WARNING
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64


# ログと出力ディレクトリの設定
LOG_DIR="./logs"
OUTPUT_DIR="./data/models/dpo"
mkdir -p $LOG_DIR $OUTPUT_DIR

# タイムスタンプの設定
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="${LOG_DIR}/dpo_output_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/dpo_error_${TIMESTAMP}.log"
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/elyza/Llama-3-ELYZA-JP-8B_DPO_${TIMESTAMP}"
mkdir -p "$MODEL_OUTPUT_DIR"

# DeepSpeed実行コマンド
deepspeed --num_gpus=4 \
    src/dpo.py \
    --deepspeed config.json \
    --model_name_or_path "elyza/Llama-3-ELYZA-JP-8B" \
    --max_length 128 \
    --epochs 1 \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --log_type wandb \
    --log_project DPO \
    --tf32 True \
    > "$OUTPUT_LOG" 2> "$ERROR_LOG"