#!/usr/bin/env bash
set -e

# 참조 대조 샘플링(auk-ref-v2): v1과 같되 지시·감정 행 절반의 참조를 반대 감정 클립으로(변수 하나). GPU 4~7.
# 기록: /home/bbungsang/projects/voice/docs/46_auk_ref_contrast.md
train_jsonl=/data/bbungsang/auk-ref-v2/train.jsonl
val_jsonl=/data/bbungsang/auk-ko-finetune/data/val.jsonl
config=ckpts/AuK/config.yaml
init_ckpt=/data/bbungsang/auk-ko-finetune/ckpt_run1/model_last.pt
output_dir=/data/bbungsang/auk-ref-v2/ckpt
num_gpus=4

learning_rate=2e-5
max_grad_norm=1.0
num_train_epochs=8
gradient_accumulation_steps=1
warmup_steps=100
frames_threshold=2700
max_samples=8
ema_beta=0.9999
ema_update_after_step=100
ema_update_every=10
save_per_updates=16535
last_per_updates=10000000
logging_steps=50
val_per_updates=16535
dataloader_num_workers=4
seed=666

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
export NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir"

accelerate launch --main_process_port 29517 --num_processes ${num_gpus} --num_machines 1 --mixed_precision bf16 \
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
