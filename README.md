# Layer-Depth-Attention

This repository starts with a minimal experiment scaffold for a decoder-only Transformer baseline.

## First experiment

The first runnable experiment is a tiny synthetic associative-recall task:

- input: a sequence of key-value pairs, then a query key
- objective: predict the value paired with the query key
- purpose: provide a cheap baseline before integrating Layer-Depth Memory Attention

## Local structure

- `train_assoc_recall.py`: training entrypoint
- `src/layer_depth_attention/data.py`: synthetic dataset generator
- `src/layer_depth_attention/model.py`: tiny decoder-only Transformer baseline

## Remote run

On the Windows server:

```bat
call D:\Annaconda\Scripts\activate.bat pt-3.9
cd /d D:\Projects\Layer-Depth-Attention
python train_assoc_recall.py --device cuda --steps 200
```
