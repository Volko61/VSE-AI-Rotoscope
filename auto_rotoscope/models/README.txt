SAM 2.1 model files go here.

Expected layout (created by scripts/fetch_assets.py):

  sam2.1_hiera_tiny/
    encoder.onnx
    decoder.onnx

These are the pre-converted SAM 2.1 "tiny" ONNX graphs (Apache-2.0). Only the
tiny variant is bundled so the built .zip stays under the 200 MB limit of the
Blender Extensions platform and works fully offline.

Sources for the ONNX files:
  https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models
  https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx
