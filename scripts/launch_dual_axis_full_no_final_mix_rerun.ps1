$ErrorActionPreference = 'Stop'
Set-Location 'D:\Projects\Layer-Depth-Attention'
if (!(Test-Path 'artifacts')) { New-Item -ItemType Directory -Path 'artifacts' | Out-Null }
if (!(Test-Path 'artifacts\logs')) { New-Item -ItemType Directory -Path 'artifacts\logs' | Out-Null }
$py = 'D:\Annaconda\envs\pt-3.9\python.exe'
$stdout = 'D:\Projects\Layer-Depth-Attention\artifacts\logs\dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun.log'
$stderr = 'D:\Projects\Layer-Depth-Attention\artifacts\logs\dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun.err.log'
$args = @(
  'scripts\train_wikitext_lm.py',
  '--device','cuda',
  '--model-preset','none',
  '--seq-len','512',
  '--d-model','384',
  '--num-layers','6',
  '--num-heads','8',
  '--mlp-ratio','4',
  '--dropout','0.2',
  '--batch-size','8',
  '--grad-accum-steps','2',
  '--steps','20000',
  '--eval-interval','400',
  '--eval-batches','20',
  '--lr','3e-4',
  '--min-lr-scale','0.1',
  '--warmup-steps','100',
  '--weight-decay','0.01',
  '--attention-type','dual_axis_full_no_final_mix',
  '--attn-residual','on',
  '--ffn-residual','on',
  '--log-backend','swanlab',
  '--log-project','Layer-Depth-Attention',
  '--log-experiment-name','dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun',
  '--run-note','rerun_same_arch_ff11d28_20260403',
  '--output','artifacts\wikitext2_dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun.json'
)
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory 'D:\Projects\Layer-Depth-Attention' -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output ("STARTED_PID=$($p.Id)")
Write-Output ("STDOUT=$stdout")
Write-Output ("STDERR=$stderr")
