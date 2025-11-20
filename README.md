# sliding_tile_attention_rocm
sliding_tile_attention for rocm

# Origin github repo
[FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/main/csrc/attn)

# For H100
Recommended Docker Containers
- ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest
- nvcr.io/nvidia/pytorch:25.01-py3(with nvcc12.8, higher version will cause some compile errors)

## performance(H100)
| Case        | flex_attention(ms) | tk kernel(ms) | triton kernel(ms) |
| :---------- | :--------- | :---------- | :---------- |
| B=2, H=24, N=69120, D=128, kernel=[3, 6, 10] | 302.23 | 206.48 | 504.24 |
| B=2, H=24, N=82944, D=128, kernel=[3, 3, 6]  | 108.11 | 91.23  | 184.32 |

# For MI3XX
## Triton
### implement
[sta_triton.py](attn/triton/sta_triton.py)

### verify accuracy
[test_sta_triton.py](attn/tests/test_sta_triton.py)

### benchmark
[bench_sta_triton.py](attn/benchmarks/bench_sta_triton.py)

## Hip (not yet)

## HipKitten(not yet)
[HipKitten](https://github.com/HazyResearch/HipKittens)

## performance(MI300X)
| Case        | flex_attention(ms) |  triton kernel(ms) |
| :---------- | :--------- | :---------- |
| B=2, H=24, N=69120, D=128, kernel=[3, 6, 10] | 845.12 | 654.74 |
| B=2, H=24, N=82944, D=128, kernel=[3, 3, 6]  | 227.84 | 238.15 |


## performance(MI350X)
| Case        | flex_attention(ms) |  triton kernel(ms) |
| :---------- | :--------- | :---------- |
| B=2, H=24, N=69120, D=128, kernel=[3, 6, 10] | 449.60 | 290.92 |
| B=2, H=24, N=82944, D=128, kernel=[3, 3, 6]  | 200.46 | 104.71 |