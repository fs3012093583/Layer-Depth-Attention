@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist logs mkdir logs
call D:\Annaconda\Scripts\activate.bat pt-3.9
python -u scripts\train_wikitext103_ablation.py ^
  --methods shared_kv_depth_memory_dualq_sublayer ^
  1>> logs\wikitext103_dualq_sublayer.log ^
  2>> logs\wikitext103_dualq_sublayer.err.log
