"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important
operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention.
And we also remove the multi-head attention part. So in total we only have two dimensions: batch
size B and sequence length S, which are aligned with N, M in the previous puzzle. After such
simplification, you will find we are not so far from the FlashAttention algorithm. And with
TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: Tensor([B, S], float32)  # input tensor
    K: Tensor([B, S], float32)  # input tensor
    V: Tensor([B, S], float32)  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384

Output:
    O: Tensor([B, S], float32)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row
    QK: Tensor([B, S], float32)  # results of q*k
    P:  Tensor([B, S], float32)  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    # TODO: Implement this function
    with T.Kernel(B // BLOCK_B, threads=256) as (bx,):  
        row_idx = bx * BLOCK_B  

        Q_reg = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        K_reg = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_reg = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_reg = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

        # 当前块
        qk = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        exp_qk = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        cur_max = T.alloc_fragment((BLOCK_B,), dtype)
        cur_sum = T.alloc_fragment((BLOCK_B,), dtype)  

        # 整行
        final_max = T.alloc_fragment((BLOCK_B,), dtype)  
        final_sum = T.alloc_fragment((BLOCK_B,), dtype)

        T.fill(final_max, -T.infinity(dtype))
        T.clear(final_sum)

        for col in T.Serial(S // BLOCK_S):
            col_idx = col * BLOCK_S
            T.copy(Q[row_idx, col_idx], Q_reg)
            T.copy(K[row_idx, col_idx], K_reg)
            
            # scalar-flash-attn 是简易版本，qk不是矩阵乘，是逐元素乘
            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                qk[i, j] = Q_reg[i,j] * K_reg[i,j]
            T.reduce_max(qk, cur_max, dim=1, clear=True)

            # 调整之前的sum
            for i in T.Parallel(BLOCK_B):
                prev_max = final_max[i]
                final_max[i] = T.max(final_max[i], cur_max[i])
                final_sum[i] = final_sum[i] * T.exp2((prev_max - final_max[i]) * log2_e)

            # 计算exp_qk，并计算cur_sum
            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                exp_qk[i, j] = T.exp2((qk[i,j] - final_max[i]) * log2_e)
            T.reduce_sum(exp_qk, cur_sum, dim=1, clear=True)   

            # 计算最终的sum
            for i in T.Parallel(BLOCK_B):
                final_sum[i] += cur_sum[i]

        for col in T.Serial(S // BLOCK_S):
            col_idx = col * BLOCK_S  
            T.copy(Q[row_idx, col_idx], Q_reg)
            T.copy(K[row_idx, col_idx], K_reg)
            T.copy(V[row_idx, col_idx], V_reg)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                # Q,K
                O_reg[i,j] = (
                    T.exp2((Q_reg[i,j] * K_reg[i,j] - final_max[i]) * log2_e) /
                    final_sum[i] * V_reg[i,j]
                )
            
            T.copy(O_reg, O[row_idx, col_idx])

    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
