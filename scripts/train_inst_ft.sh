#!/usr/bin/env bash
set -e

# Stage 1 지시어 학습 테스트: v1'(태그 FT) 위에 Instruct 템플릿(지시문/감정/둘 다) 게임 정제 3만 + v1' 리플레이 3만.
# 기록: /home/bbungsang/projects/voice/docs/41_stage1_captioning_global_instruction.md
train_jsonl=/data/bbungsang/auk-inst-ft/data/train.jsonl
val_jsonl=/data/bbungsang/auk-ko-finetune/data/val.jsonl
config=ckpts/AuK/config.yaml
init_ckpt=/data/bbungsang/auk-tag-ft/ckpt_v1p/model_last.pt
output_dir=/data/bbungsang/auk-inst-ft/ckpt
num_gpus=4

learning_rate=2e-5
max_grad_norm=1.0
num_train_epochs=2
gradient_accumulation_steps=1
warmup_steps=100
frames_threshold=2700
max_samples=8
ema_beta=0.9999
ema_update_after_step=100
ema_update_every=10
save_per_updates=100000
last_per_updates=500
logging_steps=50
val_per_updates=2000
dataloader_num_workers=4
seed=666

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir"

accelerate launch --num_processes ${num_gpus} --num_machines 1 --mixed_precision bf16 \
  -m auk.train.train \
  --train_jsonl ${train_jsonl} \
  --val_jsonl ${val_jsonl} \
  --config ${config} \
  --init_ckpt ${init_ckpt} \
  --output_dir ${output_dir} \
  --learning_rate ${learning_rate} \
  --max_grad_norm ${max_grad_norm} \
  --num_train_epochs ${num_train_epochs} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --warmup_steps ${warmup_steps} \
  --frames_threshold ${frames_threshold} \
  --max_samples ${max_samples} \
  --ema_beta ${ema_beta} \
  --ema_update_after_step ${ema_update_after_step} \
  --ema_update_every ${ema_update_every} \
  --save_per_updates ${save_per_updates} \
  --last_per_updates ${last_per_updates} \
  --logging_steps ${logging_steps} \
  --val_per_updates ${val_per_updates} \
  --dataloader_num_workers ${dataloader_num_workers} \
  --seed ${seed}
