# Fast-ACVNet+ on the Radxa Cubie A7A NPU (Allwinner A733)

Stereo depth on the Radxa Cubie A7A's NPU: [Fast-ACVNet+](https://github.com/gangweiX/Fast-ACVNet) converted to Allwinner
`.nb` files, plus the tooling and notes needed to convert it yourself. On the board, a 256x192 stereo pair takes
**about 44 ms on the NPU** (int8), against about 300 ms for the same network on the CPU through ONNX Runtime.

> Status: works on my board with the vendor toolchain below. This is a community write-up, not an official Radxa/Allwinner
> project. Numbers are from one board and a small test set (see *Limitations*).

## What is in here

| Path | What |
|---|---|
| `models/` | ready-to-run `.nb` files (256x192 int8-pcq and int16, 640x384 int8-pcq) |
| `convert/` | exporter, NPU-friendly model rewrite, ONNX patcher, Acuity scripts, one-shot driver `convert.sh` |
| `runtime/` | `nb_run.cpp` (standalone VIPLite runner: timing + real inputs), `run_stereo.py` (stereo pair -> disparity picture), helpers to compare against float ONNX |

## Results (Radxa Cubie A7A, NPU at 1008 MHz, maxdisp 64, top-k 4)

| Input (WxH) | Quantization | NPU run time | FPS | mean error vs float ONNX* | `.nb` here |
|---|---|---|---|---|---|
| 256x192 | uint8 | 43.8 ms | 22.9 | 0.47 px | no (easy to build) |
| 256x192 | **int8 per-channel** | **44.3 ms** | **22.6** | 0.17 px | yes |
| 256x192 | int16 | 64.9 ms | 15.4 | 0.004 px | yes |
| 320x224 | int8 per-channel | 64.6 ms | 15.5 | 0.18 px | no |
| 640x384 | int8 per-channel | 240.8 ms | 4.15 | 0.47 px | yes |

\* Mean absolute difference between the NPU disparity and the float ONNX disparity (same network, float32) over 9 outdoor
stereo pairs, in pixels of the *model input width*. This measures how faithfully the quantized `.nb` reproduces the float
model; it is **not** accuracy against ground truth. Times are the NPU run time measured by `nb_run` around cache flush + `vip_run_network` + cache invalidate (mean of 20+
runs, very stable, p95 within ~0.5 ms of the mean); they do not include camera capture, rectification or your own pre/post-processing.

Recommendation: **int8 per-channel (`pcq`)** is as fast as uint8 and much more accurate; uint8 shows visible spurious
near-blobs. **int16** is practically lossless but about 1.5x slower. A 320x224 *int16* build compiled but failed to run on
the board ("fail to wait network finish"), so check large int16 builds before relying on them.

## Using the prebuilt models

Input/output contract (all `.nb` here):

* **Two inputs**: left image, right image. Raw **uint8 RGB, HWC**, already resized to the model size (e.g. 256x192 -> 147,456 bytes each).
  ImageNet mean/std normalisation is done **on the NPU** (do not normalise on the CPU).
* **One output**: float32 disparity, `H x W` in pixels *of the input resolution*. Depth = `fx * baseline / disparity`, with
  `fx` scaled to the input width (`fx_full * input_width / full_width`).
* Images must be rectified (a normal stereo pair; epipolar lines horizontal).

Quick test on the board. You need the Allwinner VIPLite SDK files (header + `libNBGlinker.so` / `libVIPhal.so`), for example from
the AWNPU model zoo that goes with the Allwinner/Radxa NPU toolchain: `common/npuruntime/include` and
`common/npuruntime/lib_linux_aarch64/A733`. They are proprietary and not included here. `nb_run` calls the VIPLite API
directly, no wrapper library is needed.

```bash
cd runtime
NPU_INC=<zoo>/common/npuruntime/include NPU_LIB=<zoo>/common/npuruntime/lib_linux_aarch64/A733 ./build_on_board.sh   # -> ./nb_run
./nb_run -m ../models/fastacv_plus_md64_k4_256x192_int8pcq.nb -n 50          # timing with random input

# real images: resizes, converts to RGB, runs the NPU, saves a disparity picture (needs python3 + OpenCV)
python3 run_stereo.py --model ../models/fastacv_plus_md64_k4_256x192_int8pcq.nb --pair frame.jpg --out result
#   --pair is a side-by-side (left | right) rectified image; or use --left l.png --right r.png
#   optional: --fx <focal px at the original width> --baseline <metres>  to print depth statistics
# -> result_disp.npy (float32 disparity), result_vis.png (left | colourised disparity)
```

`nb_run` options: `-n N` timed runs, `-t S` run for S seconds, `-hz R` pace at R runs/s, `-iK file` raw input K, `-o file`
write output 0 as float32. It prints min/mean/p50/p95/p99/max latency.

To check a model against the float ONNX on your own images, prepare inputs/reference on the PC with
`runtime/prep_inputs.py`, run `runtime/run_on_board.sh` on the board, and compare with `runtime/compare_eval.py`.

### Runtime notes

* `nb_run.cpp` uses only the public VIPLite calls (`vip_init`, `vip_create_network`, `vip_create_buffer`, `vip_prepare_network`,
  `vip_set_input/output`, `vip_run_network`, `vip_flush_buffer`). It builds with `--disable-new-dtags` so that the library
  path also covers `libVIPhal.so` (a dependency of `libNBGlinker.so`); alternatively set `LD_LIBRARY_PATH`.
* The NPU has one compute core and no preemption. If another model (e.g. a detector) runs on it at the same time, the two
  simply take turns and their latencies add up.

## Converting it yourself

Prerequisites: the Allwinner/Radxa NPU toolchain docker image (`ubuntu-npu:v2.0.x`, Acuity 6.30), the
[upstream Fast-ACVNet repo](https://github.com/gangweiX/Fast-ACVNet) and its
[Fast-ACVNet+ pretrained weights](https://drive.google.com/drive/folders/1lcyzoKlkYoDL3tiPGCR6nob9WsusaTI8) (not redistributed here),
a few dozen stereo frames from your own camera for calibration, and `torch onnx onnxsim onnxruntime opencv-python` on the host.

```bash
# 1. container with a host directory mounted at /workspace, and this repo inside that directory
docker run --ipc=host -itd -v ~/npu_work:/workspace --name allwinner ubuntu-npu:v2.0.10.1 /bin/bash
git clone https://github.com/<you>/fast-acvnet-radxa-npu ~/npu_work/fast-acvnet-radxa-npu
git clone https://github.com/gangweiX/Fast-ACVNet ~/Fast-ACVNet

# optional: lets the exporter check the NPU rewrite against the original model
sed -i 's/k = 24/k = getattr(self, "topk", 24)/' ~/Fast-ACVNet/models/Fast_ACV_plus.py

# 2. one-shot conversion (export -> onnxsim -> patch -> calibration set -> Acuity import/quantize/export)
cd ~/npu_work/fast-acvnet-radxa-npu
convert/convert.sh --upstream ~/Fast-ACVNet --ckpt sceneflow.ckpt --calib-images ~/my_stereo_frames \
    --mount ~/npu_work --container allwinner --size 256x192 --maxdisp 64 --q "pcq i16"
# -> ~/npu_work/work_md64_k4_256x192/fastacv_plus_md64_k4_256x192_{pcq,i16}.nb
```

`--calib-images` is a folder of side-by-side stereo images (left | right). Use frames that look like what you will run on
(about 50, varied lighting). Sizes must be multiples of 32; `maxdisp` a multiple of 32; `topk <= maxdisp/4`.
Add `--sg-docker` if you have to call docker through `sg docker -c`.

## What had to change, and why

The stock model does not compile for the A733 NPU as is. These are the problems and their fixes (all in `convert/`):

1. **Wrong chip target.** Use `--optimize VIP9000NANODI_PLUS_PID0X1000003B` for the A733. Another target
   (`VIP9A00NANOQI_PID0X10001029`) gave empty `.nb` files and "Fatal model generation error".
2. **ScatterND/ScatterElements** (from the in-place slice assignments building the correlation volume) are rejected
   ("data type not support: FLOAT16, INT32, ASYM UINT8"). `npu_model.py` builds the volume with pad + stack instead.
3. **TopK / GatherElements** (attention top-k selection and the final top-2 regression) are rejected the same way, and
   uint8/hybrid-float quantization did not get around it. `npu_model.py` replaces them with compare + cumsum + masked sums
   (ties go to the lower index, like a stable sort). The result matches the original model to about 1e-5 in float.
4. **ONNX hygiene for Acuity's importer** (`onnx_patch.py`, after `onnxsim`): constant-fold shape arithmetic, rewrite
   `Resize` to the (X, roi, scales) form, turn `Unsqueeze`/`Squeeze` into constant `Reshape`, and add `kernel_shape`
   to `Conv`/`ConvTranspose` (newer PyTorch exporters omit it).
5. **No `torch` in the container**, so Acuity's `pegasus.py` crashes on import. `pegasus_patched.py` stubs the module
   and then runs `pegasus.py` unchanged.
6. **Preprocessing on the NPU**: `add_preproc_node: true`, `reverse_channel: false`, mean `[123.675,116.28,103.53]`,
   scale `[0.017125,0.017507,0.017429]` (ImageNet); and `add_postproc_node: true` so the output is float32.

## Limitations

* The weights are the upstream **SceneFlow** checkpoint with `maxdisp 64`, `top-k 4` (the upstream paper's configuration is
  `maxdisp 192`, `k 24`). I did not fine-tune, and I did not measure accuracy against ground-truth depth: the
  "error" column only says how closely the `.nb` matches the float model.
* Quantization used about 50 outdoor frames from one camera. Different scenes may want a different calibration set.
* Disparity range is in input-resolution pixels, so a larger input covers a smaller depth range at the same `maxdisp`.
* Tested only on the Cubie A7A (Debian 11, kernel 5.15.147-a733, VIPLite driver 2.0.3.2) and Acuity 6.30.22.

## Credits and license

Fast-ACVNet: Gangwei Xu et al., *Attention Concatenation Volume for Accurate and Efficient Stereo Matching*, CVPR 2022
([paper](https://arxiv.org/abs/2209.12699), [code](https://github.com/gangweiX/Fast-ACVNet), MIT). The `.nb` files are derived from
their released weights. NPU toolchain and runtime: Allwinner / Radxa. Everything in this repository is MIT licensed, see `LICENSE`.
