#!/usr/bin/env python3
"""Make a (simplified) Fast-ACVNet ONNX importable by Acuity 6.30.

  onnxsim model.onnx model_sim.onnx
  python convert/onnx_patch.py model_sim.onnx model_patched.onnx

Fixes applied (each one fixed a concrete Acuity import error):
  1. Resize: replace the constant `sizes` input by an explicit `scales` input, 3-input (X, roi, scales) form.
  2. Unsqueeze/Squeeze -> Reshape with a constant shape (Acuity mis-imported an Unsqueeze after ReduceMean).
  3. Add `kernel_shape` to Conv/ConvTranspose (newer torch exporters omit it; Acuity requires it).
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper as nh


def shapes_of(model):
    mi = onnx.shape_inference.infer_shapes(model)
    return {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}


def fix_resize(m):
    g = m.graph
    shp = shapes_of(m)
    for n in g.node:
        if n.op_type == "Resize" and len(n.input) > 3 and n.input[3]:
            sizes = nh.to_array([i for i in g.initializer if i.name == n.input[3]][0])
            ish = shp[n.input[0]]
            scales = np.array([sizes[i] / ish[i] for i in range(4)], dtype=np.float32)
            g.initializer.append(nh.from_array(scales, n.name + "_scales"))
            g.initializer.append(nh.from_array(np.zeros([0], np.float32), n.name + "_roi"))
            n.input[1] = n.name + "_roi"
            n.input[2] = n.name + "_scales"
            del n.input[3]
            print("Resize", n.name, ish, "->", sizes.tolist(), "scales", scales.tolist())


def squeeze_to_reshape(m):
    g = m.graph
    shp = shapes_of(m)
    cnt = 0
    for n in list(g.node):
        if n.op_type in ("Unsqueeze", "Squeeze"):
            o = shp.get(n.output[0])
            if not o or 0 in o:
                continue
            g.initializer.append(nh.from_array(np.array(o, dtype=np.int64), n.name + "_rshape"))
            n.op_type = "Reshape"
            del n.attribute[:]
            del n.input[1:]
            n.input.append(n.name + "_rshape")
            cnt += 1
    print("Unsqueeze/Squeeze -> Reshape:", cnt)


def add_kernel_shape(m):
    g = m.graph
    init = {i.name: i for i in g.initializer}
    cnt = 0
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose") and not any(a.name == "kernel_shape" for a in n.attribute):
            n.attribute.append(onnx.helper.make_attribute("kernel_shape", list(init[n.input[1]].dims[2:])))
            cnt += 1
    print("kernel_shape added:", cnt)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    m = onnx.load(src)
    fix_resize(m)
    squeeze_to_reshape(m)
    add_kernel_shape(m)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print("saved", dst)
