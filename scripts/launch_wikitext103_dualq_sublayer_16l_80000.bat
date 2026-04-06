@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist logs mkdir logs
call D:\Annaconda\Scripts\activate.bat pt-3.9
python -u scripts\train_wikitext103_ablation.py ^
  --num-layers 16 ^
  --steps 80000 ^
  --methods shared_kv_depth_memory_dualq_sublayer ^
  1>> logs\wikitext103_dualq_sublayer_16l_80000.log ^
  2>> logs\wikitext103_dualq_sublayer_16l_80000.err.log
