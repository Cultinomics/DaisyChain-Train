"""DaisyChain Task that trains the REAL SpikeWhale model on streamed FineWeb-Edu.

No reimplementation: it imports your actual `model_v2.SpikeWhaleLM` + `config.
SpikeWhaleConfig` + `spike_tokenizer.SpikeTokenizer`, builds a size chosen by the
sliders (env vars), and streams FineWeb-Edu for data. DaisyChain then distributes
it across machines; matmuls can be routed through the verified/GUDA units.

Point DaisyChain at it:
    DAISY_TASK=daisychain.spikewhale_task:SpikeWhaleTask  daisychain-train

Sliders (env):
  DAISY_SW_PATH     folder holding model_v2.py/config.py/tokenizer.json
  DAISY_SW_HIDDEN   hidden_size            (default 256)
  DAISY_SW_LAYERS   num_hidden_layers      (default 4)
  DAISY_SW_HEADS    num_attention_heads    (default 4)
  DAISY_SW_EXPERTS  n_routed_experts       (default 4)
  DAISY_SW_SEQLEN   sequence length        (default 128)
  DAISY_SW_MTP      MTP heads (0=off)      (default 0)
  DAISY_SW_DATASET  HF dataset             (default HuggingFaceFW/fineweb-edu)
  DAISY_SW_SUBSET   dataset config name    (default sample-10BT)
"""
import os
import sys

import torch

#: Where model_v2.py / config.py / tokenizer.json live. The default used to be an
#: absolute path on the author's own desktop, which in a published repo means every
#: other user gets an ImportError naming a directory that was never theirs.
_DEF_PATH = os.environ.get("DAISY_SW_PATH", os.path.join(os.getcwd(), "Spikewhale"))


def _import_spikewhale():
    if _DEF_PATH not in sys.path:
        sys.path.insert(0, _DEF_PATH)
    from config import SpikeWhaleConfig
    from model_v2 import SpikeWhaleLM
    from spike_tokenizer import SpikeTokenizer
    return SpikeWhaleConfig, SpikeWhaleLM, SpikeTokenizer


def _envi(k, d):
    return int(os.environ.get(k, d))


def build_config():
    SpikeWhaleConfig, _, _ = _import_spikewhale()
    hidden = _envi("DAISY_SW_HIDDEN", 256)
    return SpikeWhaleConfig(
        hidden_size=hidden,
        num_hidden_layers=_envi("DAISY_SW_LAYERS", 4),
        num_attention_heads=_envi("DAISY_SW_HEADS", 4),
        head_dim=32, qk_rope_head_dim=16,
        q_lora_rank=max(32, hidden // 4), o_lora_rank=max(32, hidden // 8),
        num_key_value_heads=1,
        max_position_embeddings=max(256, _envi("DAISY_SW_SEQLEN", 128)),
        moe_intermediate_size=hidden,
        n_routed_experts=_envi("DAISY_SW_EXPERTS", 4), n_shared_experts=1,
        num_experts_per_tok=min(2, _envi("DAISY_SW_EXPERTS", 4)),
        num_hash_layers=1, hc_mult=2,
        num_nextn_predict_layers=_envi("DAISY_SW_MTP", 0),
        engram_table_size=4096, engram_compress_dim=48, engram_num_heads=2,
    )


class _FineWebStream:
    """Streams FineWeb-Edu, tokenizes, yields fixed-length token windows."""

    def __init__(self, tokenizer, seqlen, rank=0, world=1):
        self.tok, self.seqlen = tokenizer, seqlen
        self.buf = []
        ds_path = os.environ.get("DAISY_SW_DATASET", "HuggingFaceFW/fineweb-edu")
        # empty string = "use the dataset's default config" (custom datasets
        # usually have no named config; the sample-10BT default only fits fineweb-edu)
        default_subset = "sample-10BT" if ds_path == "HuggingFaceFW/fineweb-edu" else ""
        subset = os.environ.get("DAISY_SW_SUBSET", default_subset)
        self.eos = getattr(tokenizer, "eos_token_id", 1) or 1
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_path, name=subset or None, split="train", streaming=True)
            ds = ds.shard(num_shards=world, index=rank) if world > 1 else ds
            self.it = iter(ds)
            self.source = f"{ds_path}:{subset or 'default'}"
        except Exception as e:
            print(f"[spikewhale] FineWeb stream unavailable ({e}); using local fallback text", flush=True)
            self.it = None
            self.source = "local-fallback"

    def _more_tokens(self):
        if self.it is not None:
            row = next(self.it)
            text = row.get("text", "") or ""
        else:
            text = ("Education is the process of learning and acquiring knowledge. "
                    "Small models can still learn useful patterns from good data. ")
        ids = self.tok.encode(text, add_special_tokens=False)
        self.buf.extend(ids + [self.eos])

    def next_window(self):
        while len(self.buf) < self.seqlen + 1:
            self._more_tokens()
        w = self.buf[: self.seqlen + 1]
        self.buf = self.buf[self.seqlen:]
        return w


class SpikeWhaleTask:
    def __init__(self):
        SpikeWhaleConfig, SpikeWhaleLM, SpikeTokenizer = _import_spikewhale()
        self.cfg = build_config()
        tok_file = os.path.join(_DEF_PATH, "tokenizer.json")
        self.tok = SpikeTokenizer(vocab_file=tok_file)
        self.seqlen = _envi("DAISY_SW_SEQLEN", 128)
        rank = _envi("RANK", 0); world = _envi("WORLD_SIZE", 1)
        self.stream = _FineWebStream(self.tok, self.seqlen, rank, world)
        self._SpikeWhaleLM = SpikeWhaleLM
        print(f"[spikewhale] data source: {self.stream.source}", flush=True)

    def build_model(self):
        torch.manual_seed(0)                       # identical init on every node
        m = self._SpikeWhaleLM(self.cfg)
        n = sum(p.numel() for p in m.parameters())
        print(f"[spikewhale] model built: {n:,} params "
              f"(hidden={self.cfg.hidden_size}, layers={self.cfg.num_hidden_layers}, "
              f"experts={self.cfg.n_routed_experts}, seqlen={self.seqlen})", flush=True)
        return m

    def sample(self, n):
        rows = [self.stream.next_window()[:self.seqlen] for _ in range(n)]
        t = torch.tensor(rows, dtype=torch.long)   # (n, seqlen)
        return t, t                                # labels == input_ids; model shifts internally

    def loss(self, model, X, y):
        return model(input_ids=X, labels=y).loss
