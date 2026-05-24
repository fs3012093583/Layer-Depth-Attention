from __future__ import annotations

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    from flash_attn import flash_attn_func

    device = "cuda"
    dtype = torch.float16
    batch_size = 2
    seq_len = 64
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    print("flash_attn output shape:", tuple(out.shape))
    print("flash-attn verification passed")


if __name__ == "__main__":
    main()
