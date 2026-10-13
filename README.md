# FLAT: Flash Linear Attention on TPUs

Pallas kernels for training linear attention layers on Google TPUs: forward and
backward passes for **Linear Attention**, **Mamba-2** and **Gated Linear
Attention (GLA)**, plus the **short depthwise causal convolution** that most of
these layers apply before the recurrence.

On one TPU v6e chip (bf16, B=8, S=4096, 16 heads of 128), forward + backward is
1.6×, 1.5× and 3.1× faster than [FLA](https://github.com/fla-org/flash-linear-attention)
on an H100 for Linear Attention, Mamba-2 and GLA. The convolution kernel is 4.9×
faster than XLA, and a full layer training step is 2.1–2.75× faster than plain JAX.

## Install

Requires Python ≥ 3.12.

```sh
pip install -e .          # add [dev] for pytest
```

## Usage

`linear_attention_packed_jax` takes a fused `qkv` projection directly, so the
kernels read query, key and value out of it without materializing the split:

```python
from flat import linear_attention_packed_jax

# qkv: (B, S, Nq*K + Nk*K + Nv*V), laid out as query | key | value
# log_forget: None             -> Linear Attention
#             (B, S, N)        -> scalar gate per head (Mamba-2)
#             (B, S, N, K)     -> diagonal gate (GLA)
y, state = linear_attention_packed_jax(
    qkv,
    num_query_heads=Nq,
    num_key_heads=Nk,
    num_value_heads=Nv,
    key_head_dim=K,
    value_head_dim=V,
    log_forget=log_forget,
)
# y: (B, S, N*V), heads flattened
```

`log_forget` holds log-decays, so its values should be ≤ 0. The backend is picked
automatically: Pallas kernels on TPU, a plain JAX reference elsewhere. Pass
`kernel_backend=KernelBackend.jax` to force the reference.

## Tests

```sh
pytest tests
```

## License

Apache 2.0
