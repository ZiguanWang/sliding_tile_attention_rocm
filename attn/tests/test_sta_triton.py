import torch
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
attn_triton_dir = os.path.join(os.path.dirname(current_dir), "triton")
sys.path.insert(0, attn_triton_dir)
from sta_triton import triton_sliding_tile_attention

from flex_sta_ref import get_sliding_tile_attention_mask
from torch.nn.attention.flex_attention import flex_attention
from tqdm import tqdm

flex_attention = torch.compile(flex_attention, dynamic=False)


def flex_test(Q, K, V, dit_seq_shape, kernel_size, text_length):
    dit_seq_shape_mapping = {
        '30x48x80':(30, 48, 80),   # Hunyuan   115200 (6x8x8)(5*6*6)  has_text
        '36x48x48':(36, 48, 48),   # Wan       82944  (6x8x8)(6*6*6)  no_text
        '18x48x80':(18, 48, 80),   # Stepvideo 69120  (6x8x8)(3*6*10) no_text
    }
    mask = get_sliding_tile_attention_mask(kernel_size, (6, 8, 8), dit_seq_shape_mapping[dit_seq_shape], text_length, 'cuda', text_length)
    output = flex_attention(Q, K, V, block_mask=mask)

    return output


def triton_fwd_kernel_test(Q, K, V, dit_seq_shape, kernel_size, text_length):
    o = triton_sliding_tile_attention(Q, K, V, kernel_size, text_length, text_length > 0, dit_seq_shape)
    return o


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()


def check_correctness(b, h, n, d, causal, dit_seq_shape, window_size, text_length, mean, std, num_iterations=50):
    results = {
        'TRITON vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }

    for _ in range(num_iterations):
        torch.manual_seed(0)

        Q = generate_tensor((b, h, n + text_length, d), mean, std, torch.bfloat16, 'cuda')
        K = generate_tensor((b, h, n + text_length, d), mean, std, torch.bfloat16, 'cuda')
        V = generate_tensor((b, h, n + text_length, d), mean, std, torch.bfloat16, 'cuda')
        triton_o = triton_fwd_kernel_test(Q, K, V, dit_seq_shape, [window_size] * h, text_length)
        pt_o = flex_test(Q, K, V, dit_seq_shape, window_size, text_length)

        diff = pt_o - triton_o
        abs_diff = torch.abs(diff)
        results['TRITON vs FLEX']['sum_diff'] += torch.sum(abs_diff).item()
        results['TRITON vs FLEX']['max_diff'] = max(results['TRITON vs FLEX']['max_diff'], torch.max(abs_diff).item())

        torch.cuda.empty_cache()

    print("max_diff", torch.max(abs_diff).item())
    print("avg_diff", torch.sum(abs_diff).item() / (b * h * n * d * 1))

    total_elements = b * h * (n + text_length) * d * num_iterations * 1
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results


def test_attention(configurations):
    for B, H, N, D, causal, dit_seq_shape, window_size, text_length in configurations:
        print("=" * 60)
        print(f"forward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}, dit_seq_shape={dit_seq_shape}, window_size={window_size}, text_length={text_length}")

        mean = 1e-1
        std = 10

        # Run correctness check directly
        results = check_correctness(B, H, N, D, causal, dit_seq_shape, window_size, text_length, mean, std, num_iterations=50)
        assert results['TRITON vs FLEX']['avg_diff'] < 4e-6, f"Average difference: {results['TRITON vs FLEX']['avg_diff']} is too large"
        assert results['TRITON vs FLEX']['max_diff'] < 4e-2, f"Maximum difference: {results['TRITON vs FLEX']['max_diff']} is too large"
        print(f"Average difference: {results['TRITON vs FLEX']['avg_diff']}")
        print(f"Maximum difference: {results['TRITON vs FLEX']['max_diff']}")


configurations = [
    (2, 24, 69120,  128, False, '18x48x80', (3, 3, 5),  0),   # Stepvideo
    (2, 24, 69120,  128, False, '18x48x80', (3, 1, 10), 0),   # Stepvideo
    (2, 24, 82944,  128, False, '36x48x48', (3, 3, 6),  0),   # Wan
    (2, 24, 82944,  128, False, '36x48x48', (3, 1, 6),  0),   # Wan
    (2, 24, 115200, 128, False, '30x48x80', (3, 3, 6),  128), # Hunyuan
    (2, 24, 115200, 128, False, '30x48x80', (3, 3, 6),  256), # Hunyuan
]

test_attention(configurations)
