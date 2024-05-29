#!/bin/sh

set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
# BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34223
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v1_5_448_finetune_continue_lora'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ${SRUN_ARGS} \
  internvl/train/internvl_chat_finetune_v2.py \
  --model_name_or_path "/mnt/workspace/wbin/modelhub/InternVL-Chat" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/mnt/workspace/wbin/datasets/pg.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --group_by_length True \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

#  --deepspeed "zero2moe.json" \
#  --deepspeed "zero_stage2_config.json" \
#   --deepspeed "zero_stage3_config.json" \
#   --moe_enable True \
