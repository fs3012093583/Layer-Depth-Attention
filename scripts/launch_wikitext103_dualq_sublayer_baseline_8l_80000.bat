@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist logs mkdir logs
call D:\Annaconda\Scripts\activate.bat pt-3.9
python -u scripts\train_wikitext103_ablation.py ^
  --num-layers 8 ^
  --steps 80000 ^
  --methods shared_kv_depth_memory_dualq_sublayer baseline ^
  1>> logs\wikitext103_dualq_sublayer_baseline_8l_80000.log ^
  2>> logs\wikitext103_dualq_sublayer_baseline_8l_80000.err.log
