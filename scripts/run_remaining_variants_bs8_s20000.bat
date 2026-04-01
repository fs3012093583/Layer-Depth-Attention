@echo off
cd /d D:\Projects\Layer-Depth-Attention
if not exist artifacts mkdir artifacts
echo [queue] start attn_residuals_dual_axis_true_bs8_s20000
call scripts\launch_attn_residuals_dual_axis_true_bs8_s20000.bat
if errorlevel 1 exit /b %errorlevel%
echo [queue] start dual_axis_memory_true_bs8_s20000
call scripts\launch_dual_axis_memory_true_bs8_s20000.bat
if errorlevel 1 exit /b %errorlevel%
echo [queue] all remaining runs finished
