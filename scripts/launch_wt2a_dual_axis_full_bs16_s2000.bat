@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist artifacts mkdir artifacts
call D:\Annaconda\Scripts\activate.bat pt-3.9
python scripts\train_wikitext_lm.py ^
  --device cuda ^
  --model-preset wt2_standard_a ^
  --attention-type dual_axis_full ^
  --steps 2000 ^
  --batch-size 16 ^
  --log-backend swanlab ^
  --log-project Layer-Depth-Attention ^
  --log-experiment-name dual_axis_full_wt2a_bs16_s2000 ^
  --output artifacts\wikitext2_dual_axis_full_wt2a_bs16_s2000.json ^
  > artifacts\wikitext2_dual_axis_full_wt2a_bs16_s2000.log 2>&1
