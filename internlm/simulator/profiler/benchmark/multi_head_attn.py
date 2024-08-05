#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

from internlm.simulator.profiler.profiler import Timer
from internlm.utils.common import get_current_device

BENCH_TYPE = "flash_attn"


def run_fa_lat_test(micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype, warmups=2, trials=5):
    # 1, S, N, H
    def run():
        device = get_current_device()
        cu_seqlens = torch.tensor([i * seqlen for i in range(micro_bsz + 1)], dtype=torch.int32, device=device)

        tfwd, tbwd = Timer(True), Timer(True)
        q = torch.rand(
            [micro_bsz * seqlen, q_head, hidden_size // q_head],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        kv = torch.rand(
            [micro_bsz * seqlen, 2, kv_head, hidden_size // q_head],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        torch.cuda.synchronize()
        tfwd.start()
        context = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=seqlen,
            max_seqlen_q=seqlen,
            causal=True,
        )
        t_fwd = tfwd.end()

        grad = torch.randn_like(context) / 32  # avoid grad is too large.
        torch.cuda.synchronize()

        tbwd.start()
        context.backward(grad, retain_graph=True)
        t_bwd = tbwd.end()
        return t_fwd, t_bwd

    for i in range(warmups):
        run()

    t_fwds, t_bwds = 0, 0
    for i in range(trials):
        t_fwd, t_bwd = run()
        t_fwds += t_fwd
        t_bwds += t_bwd

    return t_fwds / trials, t_bwds / trials


# from .base_benchmark import UnitBench
# import math

# import torch
# from einops import rearrange
# from torch import nn

# from internlm.model.registry import benchmark_initializer
# from internlm.simulator.common import TP_SIZE_RANGE, K, get_local_rank
# from internlm.utils.common import get_current_device

# try:
#     from flash_attn.flash_attn_interface import (
#         flash_attn_qkvpacked_func,
#         flash_attn_varlen_func,
#     )
#     from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
# except ModuleNotFoundError:
#     print("import fa failed!", flush=True)
#     try:
#         from deeplink_ext.internevo_ops import FlashCrossAttention, FlashSelfAttention
#     except ModuleNotFoundError:
#         flash_attn_qkvpacked_func = None
#         FlashSelfAttention = None
#         SelfAttention = None
#         print("import dipu fa failed!", flush=True)


# @benchmark_initializer.register_module(module_name=BENCH_TYPE)

# 对于FA，我们还是用on the fly的方式 profiling，并用cache缓存中间结果
# class UnitMultiHeadAttn(UnitBench):
#     # test_loop = {
#     #     "seq_len": [
#     #         64 * K,
#     #         int(0.25 * K),
#     #         int(0.5 * K),
#     #         1 * K,
#     #         2 * K,
#     #         4 * K,
#     #         8 * K,
#     #         32 * K,
#     #         16 * K,
#     #     ],  # 256 * K, 128 * K,
#     #     "head_H": [(64, 8192), (48, 6144), (32, 4096), (40, 5120)],  # (80, 10240),
#     #     "dtype": [torch.bfloat16],
#     #     "micro_bsz": [2, 1],  # 4,
#     #     "tp_size": TP_SIZE_RANGE,
#     #     "is_fwd": [True, False],
#     # }

#     def __init__(self, seq_len, num_heads_and_hidden_dim, dtype, micro_bsz, tp_size, is_fwd) -> None:
#         q_head, kv_head, embed_dim = num_heads_and_hidden_dim
#         self.num_heads_and_hidden_dim = num_heads_and_hidden_dim
#         self.TP = tp_size
#         self.S = seq_len
#         self.N = num_heads
#         self.H = embed_dim // self.N
#         self.dtype = dtype
#         self.dtype_size = 2 if self.dtype == torch.bfloat16 else 4
#         self.B = micro_bsz
#         self.oom = False
#         self.is_fwd = is_fwd
#         self.causal = True

#         assert num_heads % self.TP == 0, "num_heads must be divisible by tp_size"
#         assert num_heads >= tp_size, f"head nums must bigger then tp_size: {tp_size}"

#         self.num_atten_head_tp = num_heads // self.TP
#         self.head_dim = self.H // num_heads
#         self.tp_embedding_dim = self.H // self.TP

#         self.packed_length = self.S * self.B
#         self.device = f"cuda:{get_local_rank()}"
#         cu_seqlens = [i * self.S for i in range(self.B + 1)]

#         weights_mem_used = self.packed_length * 3 * self.H * self.dtype_size
#         attn_activation = 11 * self.packed_length * self.H
#         mem_used = attn_activation + weights_mem_used

#         self.inner_attn = FlashSelfAttention(causal=True, softmax_scale=self.H ** (0.5), attention_dropout=0.0)

#         oom = False
#         if mem_used > 75 * 1024**3:
#             oom = True

#         # 约束1: seqlen最大不能超过256K(不含)
#         # 约束2: embed_dim在被tp切过之后若大于6144， 则packed_length不能大于256k
#         if self.packed_length >= 256 * K and (self.H / self.TP) >= 6144:
#             oom = True
#         if self.S >= 256 * K and self.B > 1:
#             oom = True
#         if self.packed_length >= 524288 and (self.H / self.TP) >= 3072:
#             oom = True
#         if self.packed_length >= 1048576 and (self.H / self.TP) >= 2048:
#             oom = True

#         if oom:
#             assert (
#                 False
#             ), f"warning : mem_used: {mem_used/1024**3:.2f} GB, seq_len: {self.S}, embed_dim: {self.H}, tp_size: {self.TP}"

#         self.qkv = torch.rand(
#             size=(self.B * self.S, 3, self.N // self.TP, self.H),
#             dtype=self.dtype,
#             device=self.device,
#             requires_grad=True,
#         )

#         self.dtype_size = self.qkv.element_size()
#         self.cu_seqlens = torch.tensor(data=cu_seqlens, dtype=torch.int32, device=self.device)
#         self.max_seqlen = self.S
#         if not self.is_fwd:
#             self.output = self.run_fwd()
#             self.grad = torch.randn_like(self.output) / 32  # avoid grad is too large.

#     def run(self):
#         if self.is_fwd:
#             self.run_fwd()
#         else:
#             self.run_bwd(self.output, self.grad)

#     def run_fwd(self):
#         context = self.inner_attn(self.qkv, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen, causal=self.causal)
#         return context

#     def run_bwd(self, output, grad):
#         output.backward(grad, retain_graph=True)

#     @staticmethod
#     def gen_store_key(micro_bsz, seq_len, num_heads_and_hidden_dim, tp_size, is_fwd):
#         _, embed_dim = num_heads_and_hidden_dim
#         tp_embedding_dim = embed_dim // tp_size
#         return f"b_{micro_bsz}_s_{seq_len}_h_{tp_embedding_dim}_fwd_{is_fwd}"

#     def complexity(self):
#         return UnitMultiHeadAttn.gen_store_key(self.B, self.S, self.num_heads_and_hidden_dim, self.TP, self.is_fwd)
#         # return f"{self.S} * {self.hidden_dim} * {self.hidden_dim}"


if __name__ == "__main__":

    micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype = 1, 4096, 4096, 32, 8, torch.bfloat16
    t_fwd, t_bwd = run_fwd(micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype)
    print(f"t_fwd: {t_fwd}, t_bwd: {t_bwd}", flush=True)
