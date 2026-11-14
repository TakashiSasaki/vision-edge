"""Pure-archive loader for full Mirage inference weights, tokenizer, and charset.

Loads all 90 non-encoder inference weights plus perm1/perm2, SentencePiece RAW
tokenizer, and CTC reader charset directly from `mirage.cact` without requiring
`mirage.safetensors` or external runtime dependencies.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from archive_encoder_weights import natural_walsh

TAG_NEEDLE3 = 0x05E12A84
ENGRAM_SLOTS = 18432
ENGRAM_SEED = 0x9E3779B9
ENGRAM_PRIME = 0x01000193
ENGRAM_ORDERS = (2, 3)
ENGRAM_HEADS = 2
N_OBJECT_REFS = 64
N_TEXT_REFS = 32
VOCAB_ENTRY = 805
CHARSET_ENTRY = 804
PERM1_ENTRY = 226
PERM2_ENTRY = 227

MIRAGE_CACT_EXPECTED_SIZE = 21_611_043
MIRAGE_CACT_EXPECTED_SHA256 = (
    '4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237'
)

CODEBOOK_SLICES = {2: slice(0, 4), 3: slice(4, 12), 4: slice(12, 28)}
SUPPORTED_LANGUAGES = ('en', 'es', 'fr', 'pt', 'de', 'it', 'nl', 'pl', 'ru', 'ar', 'zh')


@dataclass(frozen=True)
class ArchiveEntry:
    index: int
    dtype: int
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    group: int
    bits: int


def read_archive_entries(data: bytes) -> tuple[np.ndarray, list[ArchiveEntry]]:
    fields = struct.unpack_from('<48I', data, 0)
    if fields[0] != TAG_NEEDLE3 or fields[1] != 806 or fields[2] != 28:
        raise ValueError(f'Unexpected archive header fields: {fields[:3]}')
    cb_len = fields[2]
    codebook = np.frombuffer(data, '<f4', count=cb_len, offset=196).astype(np.float64)
    offset = 196 + 4 * cb_len
    min_data_offset = offset + 44 * fields[1]
    entries: list[ArchiveEntry] = []
    for i in range(fields[1]):
        dtype, ndim = data[offset], data[offset + 1]
        if dtype not in (1, 2, 3, 4) or not 0 <= ndim <= 4:
            raise ValueError(f'Invalid archive entry header at index {i}: dtype={dtype} ndim={ndim}')
        shape = struct.unpack_from('<4I', data, offset + 4)[:ndim]
        start, nbytes = struct.unpack_from('<QQ', data, offset + 20)
        group, bits = struct.unpack_from('<II', data, offset + 36)
        if start < min_data_offset or start + nbytes > len(data):
            raise ValueError(f'Archive entry {i} out of bounds: start={start} nbytes={nbytes}')
        entries.append(ArchiveEntry(i, dtype, tuple(shape), start, nbytes, group, bits))
        offset += 44
    return codebook, entries


def decode_general_cq(
    raw: bytes | memoryview,
    codebook: np.ndarray,
    bits: int,
    group: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Dequantize 2/3/4-bit CQ tensor `[rows, cols]` using Fast Walsh-Hadamard transform."""
    if bits not in CODEBOOK_SLICES:
        raise ValueError(f'Unsupported CQ bit width: {bits}')
    ng = -(-cols // group)
    padded = ng * group
    index_bytes = rows * padded * bits // 8
    norm_count = rows * ng
    if len(raw) != index_bytes + 2 * norm_count:
        raise ValueError(
            f'CQ byte count mismatch: got {len(raw)}, expected {index_bytes + 2 * norm_count}'
        )
    per = 8 // bits
    mask = (1 << bits) - 1
    packed = np.frombuffer(raw[:index_bytes], dtype=np.uint8)
    codes = ((packed[:, None] >> (np.arange(per, dtype=np.uint8) * bits)) & mask).reshape(
        rows, ng, group
    )
    norms = np.frombuffer(raw[index_bytes:], dtype='<f2').astype(np.float64).reshape(rows, ng, 1)
    cb = codebook[CODEBOOK_SLICES[bits]]
    rotated = cb[codes] * norms
    result = natural_walsh(rotated).reshape(rows, padded)[:, :cols]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite CQ result')
    return result.astype(np.float32)


def _fp16_entry_to_float32(data: bytes, entry: ArchiveEntry, shape: tuple[int, ...]) -> torch.Tensor:
    count = math.prod(shape)
    expected_bytes = count * 2
    if entry.dtype != 1 or entry.nbytes != expected_bytes:
        raise ValueError(
            f'Entry {entry.index} expected fp16 {shape} ({expected_bytes} B), got dtype={entry.dtype} nbytes={entry.nbytes}'
        )
    arr = np.frombuffer(data, dtype='<f2', count=count, offset=entry.offset).astype(np.float32)
    return torch.from_numpy(arr.reshape(shape).copy())


def _fp32_entry_to_float32(data: bytes, entry: ArchiveEntry, shape: tuple[int, ...]) -> torch.Tensor:
    count = math.prod(shape)
    expected_bytes = count * 4
    if entry.dtype != 2 or entry.nbytes != expected_bytes:
        raise ValueError(
            f'Entry {entry.index} expected fp32 {shape} ({expected_bytes} B), got dtype={entry.dtype} nbytes={entry.nbytes}'
        )
    arr = np.frombuffer(data, dtype='<f4', count=count, offset=entry.offset).astype(np.float32)
    return torch.from_numpy(arr.reshape(shape).copy())


def _cq_entry_to_float32(
    data: bytes,
    entry: ArchiveEntry,
    codebook: np.ndarray,
    rows: int,
    cols: int,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    blob = memoryview(data)[entry.offset : entry.offset + entry.nbytes]
    mat = decode_general_cq(blob, codebook, entry.bits, entry.group, rows, cols)
    arr = mat.T if transpose else mat
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


@dataclass
class MirageVocab:
    count: int
    pad: int
    eos: int
    bos: int
    unk: int
    add_dummy_prefix: bool
    byte_fallback: bool
    pieces: list[str]
    types: list[int]
    scores: list[float]
    size: int
    lang_ids: dict[str, int]

    @property
    def ref_base(self) -> int:
        return self.count

    def ref_id(self, kind: str, k: int) -> int:
        return self.ref_base + (k if kind == 'object' else N_OBJECT_REFS + k)

    def piece(self, idx: int) -> str:
        if idx >= self.count:
            k = idx - self.ref_base
            return f'<obj{k}>' if k < N_OBJECT_REFS else f'<txt{k - N_OBJECT_REFS}>'
        return self.pieces[idx]

    def decode(self, ids: list[int], show_special: bool = True) -> str:
        out: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                out.append(buf.decode('utf-8', 'replace'))
                buf.clear()

        for idx in ids:
            if idx >= self.count:
                flush()
                out.append(self.piece(idx))
                continue
            typ = self.types[idx]
            pc = self.pieces[idx]
            if typ == 4:
                m = re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', pc)
                if m:
                    buf.append(int(m.group(1), 16))
                    continue
            flush()
            if typ in (1, 2, 3):
                if show_special:
                    out.append(pc)
            else:
                out.append(pc.replace('\u2581', ' '))
        flush()
        return ''.join(out)


def parse_vocab_raw(raw: bytes, embedding_rows: int = 8299) -> MirageVocab:
    count, pad, eos, bos, unk = struct.unpack_from('<5I', raw, 0)
    add_dummy_prefix = bool(raw[20])
    byte_fallback = bool(raw[21])
    off = 24
    pieces: list[str] = []
    types: list[int] = []
    scores: list[float] = []
    for _ in range(count):
        score, typ, length = struct.unpack_from('<fBH', raw, off)
        off += 7
        pieces.append(raw[off : off + length].decode('utf-8', 'replace'))
        types.append(typ)
        scores.append(score)
        off += length
    lang_ids = {
        pc[2:-2]: i
        for i, pc in enumerate(pieces)
        if re.fullmatch(r'<\|[a-z]{2}\|>', pc)
    }
    return MirageVocab(
        count=count,
        pad=pad,
        eos=eos,
        bos=bos,
        unk=unk,
        add_dummy_prefix=add_dummy_prefix,
        byte_fallback=byte_fallback,
        pieces=pieces,
        types=types,
        scores=scores,
        size=embedding_rows,
        lang_ids=lang_ids,
    )


def ctc_collapse(labels: list[int], charset: str) -> str:
    out: list[str] = []
    prev = 0
    for lab in labels:
        if lab != 0 and lab != prev:
            idx = lab - 1
            if 0 <= idx < len(charset):
                out.append(charset[idx])
        prev = lab
    return ''.join(out)


def engram_index(tokens: list[int], t: int, oi: int, order: int, h: int) -> int:
    acc = (ENGRAM_SEED * (oi * ENGRAM_HEADS + h + 1)) & 0xFFFFFFFF
    for j in range(order):
        tok = tokens[t - j] if t - j >= 0 else 0
        acc = ((acc ^ tok) * ENGRAM_PRIME) & 0xFFFFFFFF
    acc ^= acc >> 15
    return acc % ENGRAM_SLOTS


def compute_engram_rows(
    engram_table: torch.Tensor,
    tokens: list[int],
    t: int,
) -> torch.Tensor:
    """Compute [512] engram lookup concatenation at step `t` for a single site."""
    parts: list[torch.Tensor] = []
    for oi, order in enumerate(ENGRAM_ORDERS):
        for h in range(ENGRAM_HEADS):
            tbl = engram_table[oi * ENGRAM_HEADS + h]
            if t - (order - 1) < 0:
                parts.append(torch.zeros((tbl.shape[1],), dtype=torch.float32))
            else:
                idx = engram_index(tokens, t, oi, order, h)
                parts.append(tbl[idx])
    return torch.cat(parts, dim=0)


def build_non_encoder_entry_spec() -> list[dict[str, Any]]:
    """Return authoritative exact mapping between mirage.cact entries and non-encoder tensors."""
    specs: list[dict[str, Any]] = []

    def add(entry: int, name: str, mode: str, shape: tuple[int, ...], layer: int | None = None) -> None:
        specs.append({'entry': entry, 'name': name, 'mode': mode, 'shape': shape, 'layer': layer})

    add(0, 'embedding/embedding', 'cq_id', (8299, 512))
    for l in range(8):
        base = 1 + 27 * l
        add(base + 0, 'stack/layers/block/ZCRMSNorm_0/scale', 'fp16', (512,), l)
        add(base + 1, 'stack/layers/block/self_attn/q_proj/kernel', 'cq_T', (384, 512), l)
        add(base + 2, 'stack/layers/block/self_attn/k_proj/kernel', 'cq_T', (96, 512), l)
        add(base + 3, 'stack/layers/block/self_attn/v_proj/kernel', 'cq_T', (128, 512), l)
        add(base + 4, 'stack/layers/block/self_attn/q_taps', 'fp16', (3, 384), l)
        add(base + 5, 'stack/layers/block/self_attn/k_taps', 'fp16', (3, 96), l)
        add(base + 6, 'stack/layers/block/self_attn/v_taps', 'fp16', (3, 128), l)
        add(base + 7, 'stack/layers/block/self_attn/q_norm/scale', 'fp16', (48,), l)
        add(base + 8, 'stack/layers/block/self_attn/k_norm/scale', 'fp16', (48,), l)
        add(base + 9, 'stack/layers/block/self_attn/gate_proj/kernel', 'cq_T', (512, 512), l)
        add(base + 10, 'stack/layers/block/self_attn/out_proj/kernel', 'cq_T', (512, 512), l)
        add(base + 11, 'stack/layers/block/post_attn_norm/scale', 'fp16', (512,), l)
        add(base + 12, 'stack/layers/block/attn_gate', 'fp16_scalar', (1,), l)
        add(base + 13, 'stack/layers/block/pre_hada_norm/scale', 'fp16', (512,), l)
        add(base + 14, 'stack/layers/block/hadamard_mlp/d1', 'fp16', (512,), l)
        add(base + 15, 'stack/layers/block/hadamard_mlp/d2', 'fp16', (512,), l)
        add(base + 16, 'stack/layers/block/hadamard_mlp/b2', 'fp16', (512,), l)
        add(base + 17, 'stack/layers/block/hadamard_mlp/d3', 'fp16', (512,), l)
        add(base + 18, 'stack/layers/block/hadamard_mlp/d4', 'fp16', (512,), l)
        add(base + 19, 'stack/layers/block/hadamard_mlp/w1a', 'fp16', (16, 16), l)
        add(base + 20, 'stack/layers/block/hadamard_mlp/w1b', 'fp16', (32, 32), l)
        add(base + 21, 'stack/layers/block/hadamard_mlp/w2a', 'fp16', (16, 16), l)
        add(base + 22, 'stack/layers/block/hadamard_mlp/w2b', 'fp16', (32, 32), l)
        add(base + 23, 'stack/layers/block/hadamard_mlp/w3a', 'fp16', (16, 16), l)
        add(base + 24, 'stack/layers/block/hadamard_mlp/w3b', 'fp16', (32, 32), l)
        add(base + 25, 'stack/layers/block/hadamard_mlp/cond_v', 'fp16', (512, 8), l)
        add(base + 26, 'stack/layers/block/hadamard_mlp/cond_u', 'fp16', (8, 512), l)

    # mhc stack params (entries 217..225)
    add(217, 'stack/mhc_a_pre', 'fp16', (8,))
    add(218, 'stack/mhc_a_post', 'fp16', (8,))
    add(219, 'stack/mhc_a_res', 'fp16', (8,))
    add(220, 'stack/mhc_b_pre', 'fp16', (8, 4))
    add(221, 'stack/mhc_b_post', 'fp16', (8, 4))
    add(222, 'stack/mhc_b_res', 'fp16', (8, 4, 4))
    add(223, 'stack/mhc_phi_pre', 'cq_T_per_layer', (32, 2048))
    add(224, 'stack/mhc_phi_post', 'cq_T_per_layer', (32, 2048))
    add(225, 'stack/mhc_phi_res', 'cq_T_per_layer', (128, 2048))

    # engrams (entries 228..235) and final_norm (236)
    for i in range(2):
        eb = 228 + 4 * i
        add(eb + 0, f'engrams_{i}/embedding', 'cq_reshape', (73728, 128))
        add(eb + 1, f'engrams_{i}/key_proj/kernel', 'cq_T', (512, 512))
        add(eb + 2, f'engrams_{i}/value_proj/kernel', 'cq_T', (512, 512))
        add(eb + 3, f'engrams_{i}/taps', 'fp16', (4, 512))
    add(236, 'stack/final_norm/scale', 'fp16', (512,))

    # cross attention + norms (entries 238..317)
    for l in range(8):
        cb = 238 + 10 * l
        add(cb + 0, 'stack/layers/block/cross_norm/scale', 'fp16', (512,), l)
        add(cb + 1, 'stack/layers/block/cross_attn/q_proj/kernel', 'cq_T', (384, 512), l)
        add(cb + 2, 'stack/layers/block/cross_attn/k_proj/kernel', 'cq_T', (384, 512), l)
        add(cb + 3, 'stack/layers/block/cross_attn/v_proj/kernel', 'cq_T', (512, 512), l)
        add(cb + 4, 'stack/layers/block/cross_attn/gate_proj/kernel', 'cq_T', (512, 512), l)
        add(cb + 5, 'stack/layers/block/cross_attn/out_proj/kernel', 'cq_T', (512, 512), l)
        add(cb + 6, 'stack/layers/block/cross_attn/q_norm/scale', 'fp16', (48,), l)
        add(cb + 7, 'stack/layers/block/cross_attn/k_norm/scale', 'fp16', (48,), l)
        add(cb + 8, 'stack/layers/block/post_cross_norm/scale', 'fp16', (512,), l)
        add(cb + 9, 'stack/layers/block/cross_gate', 'fp16_scalar', (1,), l)

    # region memory (entries 318, 771..773)
    add(318, 'pe_gate', 'fp16', (1,))
    add(771, 'slot_proj/kernel', 'fp16', (192, 512))
    add(772, 'slot_norm/scale', 'fp16', (512,))
    add(773, 'slot_pool/kernel', 'cq_T', (512, 512))

    # text reader (entries 774..803)
    add(774, 'reader/w0', 'fp16', (3, 3, 3, 32))
    add(775, 'reader/stage_norm_0/scale', 'fp16', (32,))
    add(776, 'reader/dw_0', 'fp16', (3, 3, 1, 32))
    add(777, 'reader/pw_0/kernel', 'cq_T', (64, 32))
    add(778, 'reader/stage_norm_1/scale', 'fp16', (64,))
    add(779, 'reader/dw_1', 'fp16', (3, 3, 1, 64))
    add(780, 'reader/pw_1/kernel', 'cq_T', (128, 64))
    add(781, 'reader/stage_norm_2/scale', 'fp16', (128,))
    add(782, 'reader/dw_2', 'fp16', (3, 3, 1, 128))
    add(783, 'reader/pw_2/kernel', 'cq_T', (384, 128))
    add(784, 'reader/stage_norm_3/scale', 'fp16', (384,))
    for b in range(4):
        rb = 785 + 4 * b
        add(rb + 0, f'reader/norm_{b}/scale', 'fp16', (384,))
        add(rb + 1, f'reader/mix_{b}', 'fp16', (1, 5, 1, 384))
        add(rb + 2, f'reader/up_{b}/kernel', 'cq_T', (768, 384))
        add(rb + 3, f'reader/down_{b}/kernel', 'cq_T', (384, 384))
    add(801, 'reader/final_norm/scale', 'fp16', (384,))
    add(802, 'reader/out/kernel', 'cq_T', (307, 384))
    add(803, 'reader/out/bias', 'fp32', (307,))

    return specs


def _allocate_stacked_buffers(specs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for item in specs:
        name = item['name']
        if item['layer'] is not None and name not in tensors:
            sh = item['shape']
            if item['mode'] == 'cq_T':
                tensors[name] = torch.empty((8, sh[1], sh[0]), dtype=torch.float32)
            elif item['mode'] == 'fp16_scalar':
                tensors[name] = torch.empty((8,), dtype=torch.float32)
            else:
                tensors[name] = torch.empty((8, *sh), dtype=torch.float32)
    return tensors


def load_full_archive_weights(cact_path: Path) -> dict[str, Any]:
    """Load non-encoder weights, vocab, charset, and perms directly from `mirage.cact`."""
    data = Path(cact_path).read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    codebook, entries = read_archive_entries(data)
    specs = build_non_encoder_entry_spec()
    weights = _allocate_stacked_buffers(specs)

    for item in specs:
        idx = int(item['entry'])
        entry = entries[idx]
        mode = str(item['mode'])
        shape = tuple(item['shape'])
        name = str(item['name'])
        layer = item['layer']

        if mode == 'fp16':
            t = _fp16_entry_to_float32(data, entry, shape)
        elif mode == 'fp16_scalar':
            t = _fp16_entry_to_float32(data, entry, shape).reshape(())
        elif mode == 'fp32':
            t = _fp32_entry_to_float32(data, entry, shape)
        elif mode == 'cq_id':
            t = _cq_entry_to_float32(data, entry, codebook, shape[0], shape[1], transpose=False)
        elif mode == 'cq_T':
            t = _cq_entry_to_float32(data, entry, codebook, shape[0], shape[1], transpose=True)
        elif mode == 'cq_reshape':
            raw_t = _cq_entry_to_float32(data, entry, codebook, shape[0], shape[1], transpose=False)
            t = raw_t.reshape(4, ENGRAM_SLOTS, 128).contiguous()
        elif mode == 'cq_T_per_layer':
            raw_t = _cq_entry_to_float32(data, entry, codebook, shape[0], shape[1], transpose=False)
            t = raw_t.reshape(8, shape[0] // 8, shape[1]).transpose(1, 2).contiguous()
        else:
            raise ValueError(f'Unsupported archive entry mode: {mode}')

        if layer is not None:
            weights[name][int(layer)] = t
        else:
            weights[name] = t

    perm1 = _fp32_entry_to_float32(data, entries[PERM1_ENTRY], (512,)).to(torch.int64)
    perm2 = _fp32_entry_to_float32(data, entries[PERM2_ENTRY], (512,)).to(torch.int64)
    charset_arr = _fp32_entry_to_float32(data, entries[CHARSET_ENTRY], (306,))
    charset = ''.join(chr(int(round(float(c)))) for c in charset_arr.tolist())
    raw_vocab_entry = entries[VOCAB_ENTRY]
    vocab_raw = data[raw_vocab_entry.offset : raw_vocab_entry.offset + raw_vocab_entry.nbytes]
    vocab = parse_vocab_raw(vocab_raw, embedding_rows=8299)

    return {
        'weights': weights,
        'perm1': perm1,
        'perm2': perm2,
        'charset': charset,
        'vocab': vocab,
        'vocab_raw': vocab_raw,
        'archive_identity': {
            'path': str(cact_path),
            'size': len(data),
            'sha256': actual_sha256,
        },
        'entry_spec': specs,
    }
