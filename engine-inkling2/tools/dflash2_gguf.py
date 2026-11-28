#!/usr/bin/env python3
"""dflash2_gguf.py SRC.safetensors OUT.gguf GGUF_PY: the DFlash 2 drafter for Inkling 2 Mini (incoai/Inkling-2-Mini-DFlash2 by Inco AI
and z-lab, Apache-2.0) as the GGUF ink2_v1 reads with --mtp (told apart from DSpark by dflash.selector_rank). Only numpy and llama.cpp's
gguf-py (GGUF_PY, build/llama.cpp/gguf-py). llama.cpp's dflash tensor names; the layers' matrices, fc and the selector's hidden
projection Q8_0 (gguf-py's quantizer, as llama.cpp's converter), the norms and the convolutions' base kernels F32, the selector's
codebooks BF16 as in the checkpoint, cut to Inkling's real vocabulary (200,058 rows). Metadata for the engine's checks: the tap ids as
in the checkpoint (5, 23, 41, 48), the mask token 200056, the soft cap 30 after the selector's correction. The same bytes as the
file our server runs (risar/v_gguf_df2.py, 27 Nov 2026)."""
import hashlib
import json
import struct
import sys

import numpy as np

SRC, OUT, GGUF_PY = sys.argv[1:4]
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

NL, N, DFF, NH, NKV, HD, WIN, BLK, RANK, TOPK, GROUP, MASK, NVU = 5, 2304, 6912, 32, 8, 128, 2048, 8, 256, 16, 16, 200056, 200058
SHA = "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee"
GG = {"fc.weight": "fc.weight", "hidden_norm.weight": "enc.output_norm.weight", "norm.weight": "output_norm.weight",
      "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
      "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
      "candidate_selector.successor_codebook": "selector_successor.weight"}
GGL = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
       "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
       "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
       "self_attn.k_norm.weight": "attn_k_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
       "mlp.down_proj.weight": "ffn_down.weight", "attention_conv.base_kernel": "attn_conv_base",
       "attention_conv.kernel_projection.weight": "attn_conv_proj.weight", "mlp_conv.base_kernel": "ffn_conv_base",
       "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight"}


def gguf_name(hf):
    if hf in GG:
        return GG[hf]
    _, i, rest = hf.split(".", 2)
    return f"blk.{i}.{GGL[rest]}"


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        head.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for k, m in head.items():
            a, b = m["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype={"BF16": np.uint16, "F32": np.float32, "F16": np.float16}[m["dtype"]])
            if m["dtype"] == "BF16":
                raw = (raw.astype(np.uint32) << 16).view(np.float32)
            out[k] = raw.astype(np.float32).reshape(m["shape"]).copy()
    return out


sha = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
if sha != SHA:
    sys.exit(f"{SRC}: SHA-256 {sha} is not incoai/Inkling-2-Mini-DFlash2's {SHA}")
W = read_safetensors(SRC)
Q = gguf.GGMLQuantizationType
w = gguf.GGUFWriter(OUT, "dflash")
A = "dflash."
w.add_string("general.name", "Inkling-2-Mini-DFlash2 (ink2 engine, Q8_0)")
w.add_string("general.license", "apache-2.0")
w.add_string("general.source.url", "https://huggingface.co/incoai/Inkling-2-Mini-DFlash2")
w.add_string("general.source.sha256", sha)
for k, v in (("block_count", NL), ("embedding_length", N), ("feed_forward_length", DFF), ("attention.head_count", NH),
             ("attention.head_count_kv", NKV), ("attention.key_length", HD), ("attention.value_length", HD),
             ("attention.sliding_window", WIN), ("block_size", BLK), ("selector_rank", RANK), ("selector_top_k", TOPK),
             ("conv_kernel_size", 2), ("conv_group_size", GROUP), ("mask_token_id", MASK), ("vocab_size", NVU)):
    w.add_uint32(A + k, v)
w.add_float32(A + "attention.layer_norm_rms_epsilon", 1e-6)
w.add_float32(A + "rope.freq_base", 1e4)
w.add_float32(A + "final_logit_softcapping", 30.0)
w.add_string(A + "selector_softcap_mode", "after_correction")
w.add_array(A + "target_layers", [5, 23, 41, 48])
names = list(GG) + [f"layers.{i}.{k}" for i in range(NL) for k in GGL]
assert sorted(names) == sorted(W), sorted(set(names) ^ set(W))
datas = []
for hf in names:
    x = W[hf].astype(np.float32)
    if "codebook" in hf:
        x, ty = x[:NVU], Q.BF16
    elif x.ndim == 1 or "base_kernel" in hf:
        ty = Q.F32
    else:
        ty = Q.Q8_0
    data = x if ty == Q.F32 else quants.quantize(x, ty)
    w.add_tensor_info(gguf_name(hf), data.shape, data.dtype, data.nbytes, ty)
    datas.append(data)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for data in datas:
    w.write_tensor_data(data)
w.close()
print(f"dflash2_gguf: {OUT} ({len(names)} tensors) from {SRC}")
