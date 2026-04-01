@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist artifacts mkdir artifacts
call D:\Annaconda\Scripts\activate.bat pt-3.9
python scripts\train_wikitext_lm.py ^
  --device cuda ^
  --model-preset none ^
  --seq-len 512 ^
  --d-model 384 ^
  --num-layers 6 ^
  --num-heads 8 ^
  --mlp-ratio 4 ^
  --dropout 0.2 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --steps 30000 ^
  --eval-interval 400 ^
  --eval-batches 20 ^
  --lr 3e-4 ^
  --min-lr-scale 0.1 ^
  --warmup-steps 100 ^
  --weight-decay 0.01 ^
  --attention-type attn_residuals ^
  --attn-residual on ^
  --ffn-residual on ^
  --log-backend swanlab ^
  --log-project Layer-Depth-Attention ^
  --log-experiment-name attn_residuals_true_bs8_s30000 ^
  --output artifacts\wikitext2_attn_residuals_true_bs8_s30000.json
