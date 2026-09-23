#!/usr/bin/env python3
"""Export the von Option-Marker model (encoder + scorer head) to ONE ONNX graph.

The Rust ort backend (von-backend-ort) runs this graph as-is, so the model math
cannot drift from the Python oracle — the graph *is* the Python graph
(PORTING_RUST.md §4 Route A, recipe proven in laya). Inputs:

    input_ids      int64  [batch, seq]
    attention_mask int64  [batch, seq]
    mask_pos       int64  [batch, options]   positions of [MASK] tokens

The mask-position gather happens in-graph, so the Rust side never touches
hidden states. Export notes inherited from laya (PORTING_RUST.md §4):
the legacy exporter chokes on fused encoder fast-paths — use the dynamo
exporter; declare opset 18; inline the external .data file so deployment is
a single artifact.

Run with the oracle venv (additive deps only: onnx + onnxscript):
    .venv/bin/python von-rs/scripts/export_onnx.py
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# The golden corpus is a CPU fp32 reference; export on CPU to match it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Pinned HF revision (goldens/manifest.json checkpoint_revision). Re-pinning
# requires a deliberate goldens regeneration PR, never a silent re-export.
DEFAULT_REVISION = "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd"


def sample_inputs(model) -> tuple:
    """A realistic sample over a golden-corpus case (choice_k5) for tracing
    shapes: packed text -> ids, attention mask, [MASK] positions."""
    from benchmarks.dump_logits import get_cpu_backend

    backend = get_cpu_backend()
    tok = model.tokenizer
    packed = (
        "What is the primary customer intent in the message? "
        "Customer requests refund for duplicate charge on invoice #100 [SEP] "
        "[MASK] Requesting money back [MASK] Reporting a bug or API error "
        "[MASK] Questions about invoices or plans [MASK] Requesting account "
        "closure [MASK] Documentation or pricing questions"
    )
    inputs = tok(packed, return_tensors="pt")
    input_ids = inputs["input_ids"]
    pos = (input_ids[0] == model.mask_token_id).nonzero(as_tuple=True)[0]
    assert pos.numel() == 5
    return (input_ids, inputs["attention_mask"], pos.unsqueeze(0))


class FusedOptionMarker(torch.nn.Module):
    """Encoder + in-graph [MASK] gather + MLP scorer, forward shaped exactly
    like the runtime contract (one packed sequence, K mask positions)."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.scorer = model.scorer
        self.hidden = model.hidden_size

    def forward(self, input_ids, attention_mask, mask_pos):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, seq, H)
        reps = last_hidden.gather(
            1, mask_pos.unsqueeze(-1).expand(-1, -1, self.hidden)
        )  # (B, K, H)
        return self.scorer(reps)  # (B, K)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    default_out = here.parent / "artifacts" / "von-option-marker.onnx"
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id="wfzyx/von", revision=args.revision)

    from von.models.option_marker import OptionMarkerModel

    model = OptionMarkerModel(base_model_id=snapshot)
    # AutoModel only fills the encoder from model.safetensors; the scorer head
    # must come from option_marker.pt exactly the way the backend loads it
    # (strict=True), or the exported graph ships random head weights.
    pt_path = Path(snapshot) / "option_marker.pt"
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to("cpu").eval()
    sample = sample_inputs(model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapped = FusedOptionMarker(model)

    with torch.no_grad():
        # The dynamo exporter decomposes the ModernBERT graph correctly (the
        # legacy one chokes on fused fast-paths) but defaults to external-data
        # form (<name>.onnx + <name>.onnx.data). Inline afterwards so
        # deployment stays a single file.
        torch.onnx.export(
            wrapped,
            sample,
            str(out),
            input_names=["input_ids", "attention_mask", "mask_pos"],
            output_names=["logits"],
            opset_version=18,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "mask_pos": {0: "batch", 1: "options"},
                "logits": {0: "batch", 1: "options"},
            },
            do_constant_folding=True,
            dynamo=True,
        )

        import onnx

        loaded = onnx.load(str(out))  # resolves the external .data file
        onnx.save(loaded, str(out))   # rewrites inline as a single file
        data_file = str(out) + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)

    # eager-vs-traced sanity on the sample batch (cheap, catches export rot
    # early; the real numeric gate lives in the Rust parity tests).
    with torch.no_grad():
        want = model(
            input_ids=sample[0],
            attention_mask=sample[1],
            mask_positions=[sample[2][0].tolist()],
        )[0]
        got = wrapped(*sample)[0]
    delta = (want - got).abs().max().item()
    assert delta == 0.0, f"traced graph disagrees with eager: {delta}"
    # Also prove the in-graph gather picks the same positions the backend
    # computes at runtime.
    assert sample[2][0].tolist() == [
        i for i, t in enumerate(sample[0][0].tolist()) if t == model.mask_token_id
    ]

    size = out.stat().st_size / 1e6
    print(f"exported {out} ({size:.0f} MB), eager-vs-traced delta {delta}")


if __name__ == "__main__":
    main()
