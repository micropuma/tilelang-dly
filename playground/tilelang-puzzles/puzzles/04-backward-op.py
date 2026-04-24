"""
Puzzle 04: Backward Op
==============
This puzzle implements a backward operator for better understanding how TileLang
handles a cutomized need.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import test_puzzle

"""
Consider the fused vector multiplication ReLU example from the previous puzzle.
We now extend the first input A to be a 2D tensor (Then B is like "broadcast" to this 2D shape).

04-1: Fused multiplication ReLU with broadcasting.

Inputs:
    A: Tensor([N, M], float16)  # input tensor
    B: Tensor([M,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192

Output:
    C: Tensor([N, M], float16)  # output tensor

Definition:
    for i in range(N):
        for j in range(M):
            C[i, j] = max(0, A[i, j] * B[j])
"""


def ref_mul_relu_bcast(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[1] == B.shape[0]  # M
    assert A.dtype == B.dtype == torch.float16

    # torch.mul will automatically broadcast B to A's shape
    return (A * B).relu_()


@tilelang.jit
def tl_mul_relu_bcast(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=128) as (bx, by):
        # 用fragment限制控制一下寄存器的加速  
        idx_x = bx * BLOCK_N
        idx_y = by * BLOCK_M

        A_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_reg = T.alloc_fragment((BLOCK_M,), dtype)
        C_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        T.copy(A[idx_x, idx_y], A_reg)
        T.copy(B[idx_y], B_reg)

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):  
            C_reg[i, j] = T.max(A_reg[i, j] * B_reg[j], 0)  

        T.copy(C_reg, C[idx_x, idx_y])

    return C


def run_mul_relu_bcast():
    print("\n=== Fused Multiplication ReLU with Broadcasting ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 64
    BLOCK_M = 64
    test_puzzle(
        tl_mul_relu_bcast,
        ref_mul_relu_bcast,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


"""
Now let's consider the backward of the above operation.
We will compute the gradient of the loss w.r.t. A. So the dC is given and we
need to compute dA. According to the chain rule, our computation task can be
formalized as:

04-2: Backward of fused multiplication ReLU with broadcasting.

Inputs:
    A: Tensor([N, M], float16)  # input tensor
    B: Tensor([M,], float16)  # input tensor
    dC: Tensor([N, M], float16)  # derivative w.r.t. C
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192

Output:
    dA: Tensor([N, M], float16)  # derivative w.r.t. A

Definition:
    for i in range(N):
        for j in range(M):
            dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0)
"""


def ref_mul_relu_bwd(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[0] == dC.shape[0]  # N
    assert A.shape[1] == B.shape[0] == dC.shape[1]  # M
    assert len(dC.shape) == 2
    assert A.dtype == B.dtype == dC.dtype == torch.float16

    A = A.clone()
    B = B.clone()
    A.requires_grad_(True)
    B.requires_grad_(True)
    C = torch.relu(A * B)
    C.backward(dC)
    return A.grad


# 禁用 WARP_SPECIALIZED, TMA等高级优化，保证最小可用
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_bwd(A, B, dC, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    dC: T.Tensor((N, M), dtype)
    dA = T.empty((N, M), dtype)

    # TODO: Implement this function
    # dA = dC * B if (A*B) > 0  
    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (bx, by): # type: ignore
        idx = bx * BLOCK_N  
        idy = by * BLOCK_M  

        A_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_reg = T.alloc_fragment((BLOCK_M,), dtype)
        dC_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        dA_reg = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        T.copy(A[idx, idy], A_reg)
        T.copy(B[idy], B_reg)
        T.copy(dC[idx, idy], dC_reg)

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):  
            tmp = A_reg[i,j] * B_reg[j]
            dA_reg[i,j] = T.if_then_else(tmp > 0, dC_reg[i,j] * B_reg[j], 0)
        
        T.copy(dA_reg, dA[idx, idy])

    return dA


def run_mul_relu_bwd():
    print("\n=== Fused Multiplication ReLU with Broadcasting, Backward ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 64
    BLOCK_M = 64
    # kernel = tl_mul_relu_bwd(N, M, dtype, BLOCK_N, BLOCK_M)
    # kernel.print_source_code()
    test_puzzle(
        tl_mul_relu_bwd,
        ref_mul_relu_bwd,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


if __name__ == "__main__":
    run_mul_relu_bcast()
    run_mul_relu_bwd()
