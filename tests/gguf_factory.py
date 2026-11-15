"""Build real (tiny) GGUF files with the official gguf writer.

The files have genuine headers, tensor tables, alignment and padding, so the
parser is exercised on the real binary format rather than on mocks.
"""

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize


def build_gguf(
    path,
    architecture="llama",
    layers=2,
    embedding=64,
    heads=4,
    kv_heads=2,
    context=4096,
    file_type=None,
    key_length=None,
    value_length=None,
    kv_heads_array=None,
    experts=0,
    experts_used=0,
    sliding_window=None,
    extra_kv=None,
    tokens=0,
    parameter_count=None,
    split=None,
    tensors=None,
):
    writer = GGUFWriter(str(path), architecture)

    writer.add_name("test-model")
    writer.add_block_count(layers)
    writer.add_context_length(context)
    writer.add_embedding_length(embedding)
    writer.add_head_count(heads)

    if kv_heads_array is not None:
        writer.add_array(f"{architecture}.attention.head_count_kv",
                         kv_heads_array)
    elif kv_heads is not None:
        writer.add_head_count_kv(kv_heads)

    if key_length is not None:
        writer.add_key_length(key_length)

    if value_length is not None:
        writer.add_value_length(value_length)

    if experts:
        writer.add_expert_count(experts)
        writer.add_expert_used_count(experts_used)

    if sliding_window:
        writer.add_sliding_window(sliding_window)

    if file_type is not None:
        writer.add_file_type(file_type)

    if parameter_count is not None:
        writer.add_uint64("general.parameter_count", parameter_count)

    if split:
        writer.add_uint16("split.no", split[0])
        writer.add_uint16("split.count", split[1])

    for key, value in (extra_kv or {}).items():
        writer.add_array(key, value) if isinstance(value, list) \
            else writer.add_uint32(key, value)

    if tokens:
        writer.add_array("tokenizer.ggml.tokens",
                         [f"tok{i}" for i in range(tokens)])

    if tensors is None:
        tensors = default_tensors(embedding, experts)

    for name, array, qtype in tensors:
        if qtype is None:
            writer.add_tensor(name, array)
        else:
            writer.add_tensor(name, quantize(array, qtype),
                              raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return path


def default_tensors(embedding, experts):
    rng = np.random.default_rng(0)

    tensors = [
        ("token_embd.weight",
         rng.standard_normal((16, embedding)).astype(np.float32),
         GGMLQuantizationType.Q8_0),
        ("blk.0.attn_q.weight",
         rng.standard_normal((embedding, embedding)).astype(np.float16),
         None),
        ("output_norm.weight",
         rng.standard_normal(embedding).astype(np.float32), None),
    ]

    if experts:
        tensors.append((
            "blk.0.ffn_up_exps.weight",
            rng.standard_normal((experts, 32, embedding)).astype(np.float32),
            GGMLQuantizationType.Q8_0,
        ))

    return tensors


def bytes_reader(data):
    """read_range(start, end) over an in-memory file."""

    def read_range(start, end):
        return data[start:end]

    return read_range
