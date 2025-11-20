import math
import torch
import triton
import triton.language as tl
from typing import Tuple

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.jit
def clamp_int(value, min, max):
    ret = tl.where(value > max, max, value)
    ret = tl.where(ret < min, min, ret)
    return ret


@triton.autotune(
    configs=get_autotune_config(),
    key=['head_dim', 'img_seq_len'],
)
@triton.jit
def triton_sta_kernel(
    Q, K, V, output,
    batch_size: int, num_heads: int, seq_len: int,  head_dim: int,
    img_seq_len: int,
    text_length: int,
    canvas_t: int, canvas_h: int, canvas_w: int,
    kernel_t: int, kernel_h: int, kernel_w: int,
    tile_t: int, tile_h: int, tile_w: int,
    scale: float,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    total_tile_size = tile_t * tile_h * tile_w
    q_block_per_tile = (total_tile_size + BLOCK_Q - 1) // BLOCK_Q

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_tile_flat = tl.program_id(2) // q_block_per_tile
    q_block_idx = tl.program_id(2) % q_block_per_tile

    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DIM), dtype=tl.float32)

    q_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim
    q_base_idx = q_tile_flat * total_tile_size + q_block_idx * BLOCK_Q
    q_offset_in_tile = tl.arange(0, BLOCK_Q)
    q_idx = q_base_idx + q_offset_in_tile
    q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < total_tile_size

    q = tl.load(
        Q + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
        mask=q_mask[:, None],
        other=0.0
    )  # [BLOCK_Q, BLOCK_DIM]

    # Scale sm_scale by log_2(e) and use 2^x instead of exp
    sm_scale = scale * 1.4426950408889634

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    tiles_per_hw = num_tiles_h * num_tiles_w

    q_tile_t = q_tile_flat // tiles_per_hw
    remaining = q_tile_flat % tiles_per_hw
    q_tile_h = remaining // num_tiles_w
    q_tile_w = remaining % num_tiles_w

    kernel_center_t = clamp_int(q_tile_t, kernel_t // 2, (num_tiles_t - 1) - kernel_t // 2)
    kernel_center_h = clamp_int(q_tile_h, kernel_h // 2, (num_tiles_h - 1) - kernel_h // 2)
    kernel_center_w = clamp_int(q_tile_w, kernel_w // 2, (num_tiles_w - 1) - kernel_w // 2)

    kv_tile_start_t = kernel_center_t - kernel_t // 2
    kv_tile_end_t = kernel_center_t + kernel_t // 2 + 1
    kv_tile_end_t = tl.where(kv_tile_end_t > num_tiles_t, num_tiles_t, kv_tile_end_t)

    kv_tile_start_h = kernel_center_h - kernel_h // 2
    kv_tile_end_h = kernel_center_h + kernel_h // 2 + 1
    kv_tile_end_h = tl.where(kv_tile_end_h > num_tiles_h, num_tiles_h, kv_tile_end_h)

    kv_tile_start_w = kernel_center_w - kernel_w // 2
    kv_tile_end_w = kernel_center_w + kernel_w // 2 + 1
    kv_tile_end_w = tl.where(kv_tile_end_w > num_tiles_w, num_tiles_w, kv_tile_end_w)

    for kv_tile_t in tl.range(kv_tile_start_t, kv_tile_end_t):
        for kv_tile_h in tl.range(kv_tile_start_h, kv_tile_end_h):
            for kv_tile_w in tl.range(kv_tile_start_w, kv_tile_end_w):
                kv_base_idx = (kv_tile_t * num_tiles_h * num_tiles_w + kv_tile_h * num_tiles_w + kv_tile_w) * total_tile_size

                for kv_block_idx in tl.range(0, total_tile_size, BLOCK_KV):
                    kv_offset_in_block = tl.arange(0, BLOCK_KV)
                    kv_idx = kv_base_idx + kv_block_idx + kv_offset_in_block
                    kv_mask = (kv_block_idx + tl.arange(0, BLOCK_KV)) < total_tile_size

                    kv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim

                    k = tl.load(
                        K + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                        mask=kv_mask[:, None],
                        other=0.0
                    )  # [BLOCK_KV, BLOCK_DIM]
                    v = tl.load(
                        V + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                        mask=kv_mask[:, None],
                        other=0.0
                    )  # [BLOCK_KV, BLOCK_DIM]

                    scores = tl.dot(q, k.T)
                    scores = scores * sm_scale

                    current_m = tl.max(scores, axis=1)
                    new_m = tl.maximum(m, current_m)
                    exp_scores = tl.math.exp2(scores - new_m[:, None])
                    current_l = tl.sum(exp_scores, axis=1)

                    # Update L <- L * exp(M - M') + L1, M <- M'
                    alpha = tl.math.exp2(m - new_m)
                    l = l * alpha + current_l
                    m = new_m

                    # Update O <- O * exp(M - M') + P @ V
                    acc = (acc * alpha[:, None] + tl.dot(exp_scores.to(v.type.element_ty), v))

    output_acc = acc / l[:, None]
    tl.store(
        output + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
        output_acc,
        mask=q_mask[:, None]
    ) # [BLOCK_Q, BLOCK_DIM]


def triton_sliding_tile_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window_size, text_length: int,
    has_text=True, dit_seq_shape='30x48x80') -> torch.Tensor:
    seq_length = q.shape[2]
    dit_seq_shape_mapping = {
        '30x48x80':1,   # Hunyuan   115200 (6x8x8)(5*6*6)  has_text
        '36x48x48':2,   # Wan       82944  (6x8x8)(6*6*6)  no_text
        '18x48x80':3,   # Stepvideo 69120  (6x8x8)(3*6*10) no_text
    }
    if has_text:
        assert q.shape[2] >= 115200 and q.shape[2] <= 115456, f"Unsupported {dit_seq_shape}, current shape is {q.shape}, only support '30x48x80' for HunyuanVideo"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)
    else:
        if dit_seq_shape == '36x48x48': # Stepvideo
            assert q.shape[2] == 82944
        elif dit_seq_shape == '18x48x80': # Wan
            assert q.shape[2] == 69120
        else:
            raise ValueError(f"Unsupported {dit_seq_shape}, current shape is {q.shape}, only support '36x48x48' for Stepvideo and '18x48x80' for Wan")
    assert q.shape[1] == len(window_size), "Number of heads must match the number of window sizes"

    batch_size, num_heads, seq_len, head_dim = q.shape
    if dit_seq_shape == '30x48x80': # Hunyuan
        canvas_t, canvas_h, canvas_w = 30, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '36x48x48': # Stepvideo
        canvas_t, canvas_h, canvas_w = 36, 48, 48
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '18x48x80': # Wan
        canvas_t, canvas_h, canvas_w = 18, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8

    # all kernel_size is the same in window_size
    kernel_size = window_size[0]
    kernel_t, kernel_h, kernel_w = kernel_size
    img_seq_len = canvas_t * canvas_h * canvas_w

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    num_tiles = num_tiles_t * num_tiles_h * num_tiles_w

    total_tile_size = tile_t * tile_h * tile_w

    # BLOCK_Q=128
    # BLOCK_KV=128
    BLOCK_DIM = head_dim

    output = torch.empty_like(q)

    # triton_sta_kernel[(batch_size, num_heads, num_tiles * triton.cdiv(total_tile_size,BLOCK_Q))](
    grid = lambda META: (batch_size, num_heads, num_tiles * triton.cdiv(total_tile_size, META['BLOCK_Q']))
    triton_sta_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        img_seq_len,
        text_length,
        canvas_t, canvas_h, canvas_w,
        kernel_t, kernel_h, kernel_w,
        tile_t, tile_h, tile_w,
        scale=1.0 / (head_dim ** 0.5),
        # BLOCK_Q=BLOCK_Q,
        # BLOCK_KV=BLOCK_KV,
        BLOCK_DIM=BLOCK_DIM,
    )

    return output
