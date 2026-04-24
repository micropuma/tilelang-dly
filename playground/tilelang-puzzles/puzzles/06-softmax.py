"""
Puzzle 06: Softmax
==============
Softmax is the first fundermental NN operator we learn in this tutorial.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Softmax operator goes a little beyond the reduce sum. We also need to use serial loop to
accumulate the summation. And we need to perform an element-wise exp operation on each element
at the same time.

Note that softmax needs to be computed in numerically stable form as in Python. To achieve this,
we need to subtract the maximum value of each row from all elements in that row
before applying the exponential function.

HINT:
1. Use `T.fill` to set the initial value of the buffer. `T.clear` sets all elements to zero by
default, which may not be what you want.

3.We recommend not using `T.exp` but instead using `T.exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

The constant log2_e is provided.

BONUS: Use "Online Softmax" algorithm to implement optimized softmax. This is also a core idea of
FlashAttention algorithm. Through this, we can implement softmax with only two passes / loops.

06-1: Softmax.

Inputs:
    A: Tensor([N, M], float32)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384

Output:
    B: Tensor([N, M], float16)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row

Definition:
    for i in range(N):
        S = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


# 朴素实现，三次pass
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    # 2的n次方更自然
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    # TODO: Implement this function
    # 1. 3次pass 
    # 2. 内层循环处理max和sum
    with T.Kernel(N // BLOCK_N, threads=256) as (bx,):
        row = bx * BLOCK_N

        A_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        max_val = T.alloc_fragment((BLOCK_N,), dtype)   
        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        row_sum = T.alloc_fragment((BLOCK_N,), dtype)

        T.fill(max_val, -T.infinity(dtype))
        T.clear(row_sum)

        # 收集最大值
        for col in T.Serial(M // BLOCK_M):  
            col_idx = col * BLOCK_M  
            T.copy(A[row, col_idx], A_reg)
            T.reduce_max(A_reg, cur_max, dim=1, clear=False)

            for row_idx in T.Parallel(BLOCK_N):
                max_val[row_idx] = T.max(max_val[row_idx], cur_max[row_idx])

        # 计算row sum
        for col in T.Serial(M // BLOCK_M):  
            col_idx = col * BLOCK_M  
            T.copy(A[row, col_idx], A_reg)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):  
                B_reg[i, j] = T.exp2((A_reg[i, j] - max_val[i]) * log2_e)
            T.reduce_sum(B_reg, row_sum, dim=1, clear=False);

        # 每个元素做处理
        for col in T.Serial(M // BLOCK_M):
            col_idx = col * BLOCK_M  
            T.copy(A[row, col_idx], A_reg)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):  
                B_reg[i, j] = T.exp2((A_reg[i, j] - max_val[i]) * log2_e)
                B_reg[i, j] = B_reg[i, j] / row_sum[i]
            T.copy(B_reg, B[row, col_idx])

    return B

# LSE实现，两次pass 


# FlashAttn实现，两次pass 
@tilelang.jit
def tl_flashattn_softmax(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    with T.Kernel(N // BLOCK_N) as (bx,):  
        row_idx = bx * BLOCK_N  
        A_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        cur_exp_a = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)
        
        row_sum = T.alloc_fragment((BLOCK_N,), dtype)
        row_max = T.alloc_fragment((BLOCK_N,), dtype)  

        for col in T.Serial(M // BLOCK_M):
            col_idx = col * BLOCK_M  
            T.copy(A[row_idx, col_idx], A_reg)
            T.reduce_max(A_reg, cur_max, dim=1, clear=True)   # 注意这里一定是clear=True，因为是局部的最大  

            # 做一下rescale操作，微调row_sum
            for row in T.Parallel(BLOCK_N):  
                row_max_prev = row_max[row]
                row_max[row] = T.max(row_max[row], cur_max[row])
                row_sum[row] = row_sum[row] * T.exp2((row_max_prev - row_max[row]) * log2_e)

            # 计算当前exp
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                cur_exp_a[i, j] = T.exp2((A_reg[i, j] - row_max[i]) * log2_e)
            # 计算当前sum 
            T.reduce_sum(cur_exp_a, cur_sum, dim=1, clear=True)    # 注意这里一定是clear=True

            # 累加row_sum
            for row in T.Parallel(BLOCK_N):  
                row_sum[row] += cur_sum[row]

        # 算好了每一行的rowsum和rowmax，做最终的处理
        for col in T.Serial(M // BLOCK_M):
            col_idx = col * BLOCK_M
            T.copy(A[row_idx, col_idx], A_reg)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                B_reg[i, j] = T.exp2((A_reg[i, j] - row_max[i]) * log2_e)
                B_reg[i, j] = B_reg[i, j] / row_sum[i]
            T.copy(B_reg, B[row_idx, col_idx])
        
    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    test_puzzle(
        tl_flashattn_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )
    bench_puzzle(
        tl_flashattn_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()
