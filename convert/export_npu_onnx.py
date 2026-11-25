#!/usr/bin/env python3
"""Export Fast-ACVNet+ to an NPU-friendly ONNX (see npu_model.py for what is rewritten).

Needs the upstream repo (https://github.com/gangweiX/Fast-ACVNet) and a Fast-ACVNet+ checkpoint.

  python convert/export_npu_onnx.py --upstream ~/Fast-ACVNet --ckpt sceneflow.ckpt \
         --maxdisp 64 --topk 4 --size 256x192 --out work/fastacv.onnx

Equivalence check: the NPU variant is compared with the original model. For that the upstream file
models/Fast_ACV_plus.py must read k from the model, i.e. change `k = 24` to `k = getattr(self, "topk", 24)`
(see README). Without that edit the check is skipped (the export itself does not depend on it).
"""
import argparse, os, sys, collections
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="path to a clone of gangweiX/Fast-ACVNet")
    ap.add_argument("--ckpt", required=True, help="Fast-ACVNet+ checkpoint (e.g. sceneflow.ckpt)")
    ap.add_argument("--maxdisp", type=int, default=64, help="multiple of 32 (maxdisp/4 must be divisible by 8)")
    ap.add_argument("--topk", type=int, default=4, help="attention top-k, must be <= maxdisp/4")
    ap.add_argument("--size", default="256x192", help="WxH, multiples of 32")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w, h = map(int, a.size.lower().split("x"))
    assert w % 32 == 0 and h % 32 == 0, "size must be multiples of 32"
    assert a.maxdisp % 32 == 0 and a.topk <= a.maxdisp // 4

    sys.path.insert(0, os.path.abspath(a.upstream))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import __models__
    from npu_model import Fast_ACVNet_plus_npu

    model = __models__["Fast_ACVNet_plus"](a.maxdisp, False)
    state = torch.load(a.ckpt, map_location="cpu")["model"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    model.topk = a.topk
    model.eval()

    x, y = torch.randn(1, 3, h, w), torch.randn(1, 3, h, w)
    patched = 'getattr(self, "topk"' in open(os.path.join(a.upstream, "models", "Fast_ACV_plus.py")).read()
    with torch.no_grad():
        ref = model(x, y)[0] if patched else None
        model.__class__ = Fast_ACVNet_plus_npu
        new = model(x, y)[0]
    if ref is not None:
        print("max |original - NPU variant| =", float((ref - new).abs().max()))
    else:
        print("equivalence check skipped (upstream Fast_ACV_plus.py not patched, see README)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (x, y), a.out, export_params=True, opset_version=a.opset,
                          do_constant_folding=True, input_names=["left_image", "right_image"],
                          output_names=["output"], dynamic_axes=None)
    import onnx
    m = onnx.load(a.out)
    onnx.save_model(m, a.out, save_as_external_data=False)      # single file
    if os.path.exists(a.out + ".data"):
        os.remove(a.out + ".data")
    print(collections.Counter(n.op_type for n in m.graph.node).most_common(8), "...")
    print("saved", a.out)


if __name__ == "__main__":
    main()
