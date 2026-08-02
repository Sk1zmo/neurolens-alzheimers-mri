"""Export the trained model to ONNX for serverless inference.

Vercel functions cannot carry PyTorch (the wheel alone blows past the 250 MB
bundle limit), so the deployed model is ONNX + onnxruntime, which lands around
40 MB all-in including numpy and Pillow.

The exported graph has two outputs:

  logits : (1, 4)        raw class scores
  cam    : (1, 4, 7, 7)  class activation maps for *every* class

`cam` is produced by folding the classifier weight matrix into a 1x1
convolution over the final feature map. Because the head is
GAP -> Linear, that 1x1 conv is mathematically the classic CAM, so the
serverless runtime gets a real, gradient-free explanation map for free — it
just picks the predicted class's channel, ReLUs it and upsamples.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from model import AlzheimerNet


class CamExportModel(nn.Module):
    def __init__(self, net: AlzheimerNet):
        super().__init__()
        self.features = net.features
        self.pool = net.pool
        self.classifier = net.classifier
        # W: (num_classes, 1280) -> 1x1 conv kernel (num_classes, 1280, 1, 1)
        self.cam_conv = nn.Conv2d(net.classifier.in_features,
                                  net.classifier.out_features, 1, bias=False)
        with torch.no_grad():
            self.cam_conv.weight.copy_(
                net.classifier.weight.detach().view(
                    net.classifier.out_features, net.classifier.in_features, 1, 1)
            )

    def forward(self, x):
        fmap = self.features(x)
        logits = self.classifier(torch.flatten(self.pool(fmap), 1))
        cam = self.cam_conv(fmap)
        return logits, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINT_DIR / "best.pt"))
    ap.add_argument("--out", default=str(C.ARTIFACTS_DIR / "model.onnx"))
    ap.add_argument("--quantize", action="store_true",
                    help="Also emit a dynamically int8-quantised copy.")
    ap.add_argument("--deploy-to", default=str(C.PROJECT_ROOT / "web" / "api" / "model"),
                    help="Directory the Vercel function reads the model from.")
    args = ap.parse_args()

    C.ensure_dirs()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net = AlzheimerNet(pretrained=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    export_model = CamExportModel(net).eval()
    dummy = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)

    out_path = Path(args.out)
    torch.onnx.export(
        export_model, dummy, str(out_path),
        input_names=["input"], output_names=["logits", "cam"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"},
                      "cam": {0: "batch"}},
        opset_version=C.ONNX_OPSET, do_constant_folding=True,
    )
    print(f"exported {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ---- parity check against PyTorch, on real images ----
    # Probing with torch.randn is worse than useless here: noise drives this
    # network to logits of +/-500, where ORT's and PyTorch's different conv
    # algorithms produce absolute gaps of ~1 that mean nothing. Validate on
    # actual preprocessed test slices — the distribution the model will serve.
    import onnxruntime as ort
    from PIL import Image
    from dataset import eval_transform

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    split = C.SPLITS_DIR / "test.json"
    if split.exists():
        recs = json.loads(split.read_text(encoding="utf-8"))
        step = max(1, len(recs) // 12)
        probes = []
        tf = eval_transform()
        for r in recs[::step][:12]:
            probes.append(tf(Image.open(r["path"]).convert("RGB")).unsqueeze(0))
        source = f"{len(probes)} held-out test slices"
    else:
        torch.manual_seed(0)
        probes = [torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE) for _ in range(4)]
        source = "random tensors (test split unavailable — weak check)"

    worst_logit = worst_prob = worst_cam = 0.0
    mismatches = 0
    for probe in probes:
        with torch.no_grad():
            t_logits, t_fmap = net.forward_with_features(probe)
            t_cam = torch.einsum("bkhw,ck->bchw", t_fmap,
                                 net.classifier.weight.detach())
        o_logits, o_cam = sess.run(None, {"input": probe.numpy()})

        t_p = torch.softmax(t_logits, dim=1).numpy()
        o_p = torch.softmax(torch.from_numpy(o_logits), dim=1).numpy()
        worst_prob = max(worst_prob, float(np.abs(o_p - t_p).max()))
        worst_logit = max(worst_logit,
                          float(np.abs(o_logits - t_logits.numpy()).max()))
        worst_cam = max(worst_cam, float(np.abs(o_cam - t_cam.numpy()).max()))
        mismatches += int(o_logits.argmax() != t_logits.numpy().argmax())

    print(f"parity on {source}:")
    print(f"  max |Δlogit| {worst_logit:.2e} | max |Δprob| {worst_prob:.2e} | "
          f"max |Δcam| {worst_cam:.2e} | argmax mismatches {mismatches}")
    if mismatches:
        raise SystemExit("ONNX export changes the predicted class — do not ship.")
    if worst_prob > 1e-3:
        raise SystemExit(
            f"ONNX probabilities drift by {worst_prob:.2e} from PyTorch.")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = out_path.with_name(out_path.stem + "_int8.onnx")
        quantize_dynamic(str(out_path), str(q_path), weight_type=QuantType.QUInt8)
        print(f"quantised {q_path}  ({q_path.stat().st_size / 1e6:.1f} MB)")

    # ---- sidecar metadata the serving function needs ----
    metrics_path = C.REPORTS_DIR / "metrics.json"
    temperature = 1.0
    headline = {}
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        temperature = m.get("temperature", 1.0)
        headline = m.get("headline", {})
    else:
        print("! metrics.json missing — run evaluate.py for a calibrated "
              "temperature. Falling back to T=1.0")

    meta = {
        "model_name": C.MODEL_NAME,
        "version": ckpt.get("version", "1.0.0"),
        "classes": C.CLASS_LABELS,
        "class_dirs": C.CLASS_DIRS,
        "img_size": C.IMG_SIZE,
        "mean": C.IMAGENET_MEAN,
        "std": C.IMAGENET_STD,
        "temperature": temperature,
        "trained_epoch": ckpt.get("epoch"),
        "val": ckpt.get("val", {}),
        "test_headline": headline,
    }
    (C.ARTIFACTS_DIR / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # ---- copy into the web project ----
    deploy = Path(args.deploy_to)
    deploy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, deploy / "model.onnx")
    shutil.copy2(C.ARTIFACTS_DIR / "model_meta.json", deploy / "model_meta.json")
    print(f"deployed model + meta -> {deploy}")

    public = C.PROJECT_ROOT / "web" / "public" / "model"
    if metrics_path.exists():
        public.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_path, public / "metrics.json")
        for png in ("confusion_matrix.png", "roc_curves.png"):
            src = C.REPORTS_DIR / png
            if src.exists():
                shutil.copy2(src, public / png)
        print(f"deployed metrics -> {public}")


if __name__ == "__main__":
    main()
