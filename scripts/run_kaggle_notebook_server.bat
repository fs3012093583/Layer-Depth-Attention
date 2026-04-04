@echo off
cd /d D:\Projects\Layer-Depth-Attention
call D:\Annaconda\Scripts\activate.bat pt-3.9
jupyter nbconvert --to notebook --execute notebooks\attention-only-ablation-notebook-server.ipynb --output attention-only-ablation-notebook-server-executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=-1 > logs\kaggle_notebook_baseline.log 2>&1
