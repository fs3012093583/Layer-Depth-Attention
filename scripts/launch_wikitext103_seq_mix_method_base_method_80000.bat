@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist logs mkdir logs
call D:\Annaconda\Scripts\activate.bat pt-3.9

echo [phase 1] method seq256 steps80000 >> logs\wikitext103_seq_mix_method_base_method_80000.log
python -u scripts\train_wikitext103_ablation.py ^
  --max-seq-len 256 ^
  --steps 80000 ^
  --methods shared_kv_depth_memory_dualq_sublayer ^
  1>> logs\wikitext103_seq_mix_method_base_method_80000.log ^
  2>> logs\wikitext103_seq_mix_method_base_method_80000.err.log

echo [phase 2] baseline seq512 steps80000 >> logs\wikitext103_seq_mix_method_base_method_80000.log
python -u scripts\train_wikitext103_ablation.py ^
  --max-seq-len 512 ^
  --steps 80000 ^
  --methods baseline ^
  1>> logs\wikitext103_seq_mix_method_base_method_80000.log ^
  2>> logs\wikitext103_seq_mix_method_base_method_80000.err.log

echo [phase 3] method seq256 steps80000 >> logs\wikitext103_seq_mix_method_base_method_80000.log
python -u scripts\train_wikitext103_ablation.py ^
  --max-seq-len 256 ^
  --steps 80000 ^
  --methods shared_kv_depth_memory_dualq_sublayer ^
  1>> logs\wikitext103_seq_mix_method_base_method_80000.log ^
  2>> logs\wikitext103_seq_mix_method_base_method_80000.err.log
