# Auto Rotoscope VSE

Automatic rotoscoping for Blender's **Video Sequence Editor**, powered by
**SAM 2.1** running on **ONNX Runtime** (no PyTorch).

Click an object in the preview, SAM 2.1 produces the mask, the selection is
**propagated** across the frame range, and you **add or remove points** to
correct it. The output is an **alpha matte PNG sequence** you can reuse in the
VSE or in compositing.

- **CPU by default**, **GPU used when available** (best-effort).
- **Fully self-contained**: no network access at any point, in line with the
  `extensions.blender.org` rules (~200 MB per ZIP, no external downloads).
- The **tiny** model is bundled — the only variant that fits under the 200 MB
  cap once ONNX Runtime and OpenCV are included. The model selector stays
  extensible for off-platform builds.

> ⚠️ SAM 2 has no ONNX export of its **video mode** (memory attention — see
> [facebookresearch/sam2#702](https://github.com/facebookresearch/sam2/issues/702)).
> Temporal tracking is therefore done with **optical flow (OpenCV)**: the mask
> and points from frame N are advected to N+1, then SAM re-refines every frame.

## Scope

- **Windows x64**, **Linux x64** and **macOS Apple Silicon** (macOS 13+),
  Blender **5.1** or newer (Python 3.13, cp313 wheels). To target Blender
  4.2–5.0 (Python 3.11), regenerate with cp311 — see `PY_TAG` in
  `scripts/fetch_assets.py` and `blender_version_min` in the manifest.
  macOS Intel is not a target: Blender itself dropped it at 5.0.
- **VSE only** (movie strips / image sequences).
- **GPU**: DirectML on Windows (NVIDIA/AMD/Intel, no CUDA install needed),
  CoreML on macOS. Linux runs on CPU — the onnxruntime-gpu/CUDA package would
  blow the size cap.

> ⚠️ Only the **Windows** package has actually been installed and run. The Linux
> and macOS packages build with the right wheels but are untested — see
> [RELEASE.md](RELEASE.md) §0 before publishing them.

**Performance.** The image encoder dominates: measured at 1080p on the dev
machine, ~**3.7 s/frame** on CPU and ~**1.2 s/frame** on DirectML; the decoder is
50–90 ms. Input is resized to 1024², so source resolution barely affects it.
This is an offline tool, not a real-time one.

## Layout

```
auto_rotoscope/            # the extension package (this is what gets zipped)
  blender_manifest.toml
  __init__.py              # register()/unregister(), keymap
  preferences.py           # device (Auto/CPU/GPU), output folder
  properties.py            # session state (+/- points, range, model)
  engine/
    ort_session.py         # execution-provider selection (DirectML → CPU)
    loader.py              # ONNX session loading and caching
    sam2.py                # SAM 2.1 image pipeline (encoder/decoder ONNX)
    maskproc.py            # single-shape cleanup, feathering, contours
    propagate.py           # optical-flow propagation (Farneback)
  ops/
    common.py              # strip frame reading, preview helpers
    op_pick.py             # modal picking (click = +, Ctrl+click = −)
    op_track.py            # interactive propagation + matte export
    op_clear.py            # point reset
    output.py              # matte PNG writing, result strip
  ui/
    panel_vse.py           # "SAM2 Roto" sidebar panel in the VSE
    overlay.py             # mask outline drawn over the preview
  models/                  # <- SAM 2.1 tiny ONNX (fetched by the script)
  wheels/                  # <- onnxruntime + opencv wheels (fetched)
  licenses/                # NOTICE + full third-party license texts
scripts/
  fetch_assets.py          # downloads models + wheels, rewrites wheels[]
  build.py                 # builds dist/*.zip and checks them against the rules
  smoke_test.py            # end-to-end check of the installed extension
  make_icon.py             # generates branding/icon_*.png for the store listing
```

## Release workflow

See **[RELEASE.md](RELEASE.md)** for the full submission walkthrough. The short
version:

```bash
python scripts/fetch_assets.py     # once — downloads model + wheels (~250 MB)
python scripts/build.py            # -> dist/*.zip, validated and size-checked

blender --command extension install-file -r user_default -e dist/auto_rotoscope-0.1.0-windows_x64.zip
blender -b --python scripts/smoke_test.py
```

`scripts/build.py` finds Blender automatically (or pass `--blender <path>` /
set `$BLENDER`), empties `dist/`, runs `extension validate` and
`extension build --split-platforms`, then verifies every ZIP: under the 200 MB
cap, correct wheels for the platform, required files present, no `__pycache__`
or nested archives.

## Using it

`Edit > Preferences > Get Extensions > (dropdown) Install from Disk…`, pick the
ZIP for your platform.

In a **Video Editing** workspace: select a movie strip, open the sidebar (`N`) →
**SAM2 Roto** tab. Then **Pick** (click the object, or press `L` in the preview),
then **Track & Export Matte**.

During tracking: `Space` play/pause, `←`/`→` step, `Enter` finish, `Esc` cancel.
Mattes are written as you go, so partial results always survive a cancel.

## Platform compliance notes

- No `network` permission — everything is bundled and works offline. The smoke
  test asserts this by poisoning `socket.socket` during a model load.
- `files` permission declared (writing the generated matte sequences).
- Plain-text Python, no bytecode, no obfuscation.
- Wheels are unmodified PyPI artifacts under `./wheels/`.
- GPL-3.0-or-later; the SAM 2.1 model is Apache-2.0 (compatible, redistributable).

## `abi3` wheel caveat

The OpenCV wheels are tagged `cp37-abi3`. Some Blender versions had trouble
recognising `abi3` tags (they expected `cp313`). If the extension rejects the
OpenCV wheel, rename `...-cp37-abi3-...` to `...-cp313-abi3-...` and update the
manifest to match. Blender 5.2 accepts the `cp37-abi3` tag as-is.

## Third-party licenses

- **SAM 2.1** — Apache-2.0 (Meta Platforms, Inc.)
- **onnxruntime / onnxruntime-directml** — MIT (Microsoft)
- **opencv-python-headless** — Apache-2.0 (OpenCV) / MIT (packaging)

Full texts in [auto_rotoscope/licenses/](auto_rotoscope/licenses/); see
[NOTICE.txt](auto_rotoscope/licenses/NOTICE.txt).
