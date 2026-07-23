"""SAM 2.1 image pipeline on onnxruntime (no PyTorch).

Two ONNX graphs are used, following the widely-used samexporter / vietanhdev
export layout:

  encoder.onnx : image [1,3,1024,1024] -> high_res_feats_0, high_res_feats_1, image_embed
  decoder.onnx : (embeddings, point_coords, point_labels, mask_input, has_mask_input)
                 -> masks (low-res logits), iou_predictions

Input/output names vary slightly between exports, so we resolve them from the
session metadata by name-substring first, then by position — this keeps the code
working across model variants without hard-coding.
"""

from __future__ import annotations

import numpy as np

INPUT_SIZE = 1024
LOW_RES = 256

# ImageNet normalization used by SAM 2.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _find(names: list[str], *needles: str) -> str | None:
    for n in names:
        low = n.lower()
        if any(needle in low for needle in needles):
            return n
    return None


class SAM2Image:
    """Holds encoder/decoder sessions and per-frame embedding cache."""

    def __init__(self, encoder_session, decoder_session):
        self.enc = encoder_session
        self.dec = decoder_session

        self._enc_in = [i.name for i in self.enc.get_inputs()]
        self._enc_out = [o.name for o in self.enc.get_outputs()]
        self._dec_in = [i.name for i in self.dec.get_inputs()]
        self._dec_out = [o.name for o in self.dec.get_outputs()]

        # Resolve encoder output names (order: hi-res 0, hi-res 1, embed).
        self._out_hr0 = _find(self._enc_out, "high_res_feats_0", "high_res_feat_0") or self._enc_out[0]
        self._out_hr1 = _find(self._enc_out, "high_res_feats_1", "high_res_feat_1") or self._enc_out[1]
        self._out_embed = _find(self._enc_out, "image_embed", "embed") or self._enc_out[-1]

        # Resolve decoder input names.
        self._in_embed = _find(self._dec_in, "image_embed", "embed")
        self._in_hr0 = _find(self._dec_in, "high_res_feats_0", "high_res_feat_0", "feats_0")
        self._in_hr1 = _find(self._dec_in, "high_res_feats_1", "high_res_feat_1", "feats_1")
        self._in_coords = _find(self._dec_in, "point_coords", "coords")
        self._in_labels = _find(self._dec_in, "point_labels", "labels")
        self._in_mask = _find(self._dec_in, "mask_input")
        self._in_has_mask = _find(self._dec_in, "has_mask_input", "has_mask")
        # Some exports require the original image size (int64 [h, w]).
        self._in_orig_size = _find(self._dec_in, "orig_im_size", "orig_size", "im_size")

    # -- encoding -----------------------------------------------------------

    @staticmethod
    def preprocess(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """image_rgb: HxWx3 uint8 (RGB). Returns (blob [1,3,1024,1024], (H,W))."""
        h, w = image_rgb.shape[:2]
        # Simple resize to a square 1024 (SAM 2 expects a fixed square input).
        import cv2

        resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        arr = (arr - _MEAN) / _STD
        return arr[None, ...].astype(np.float32), (h, w)

    def encode(self, image_rgb: np.ndarray) -> dict:
        """Run the image encoder; returns a reusable embedding dict for one frame."""
        blob, (h, w) = self.preprocess(image_rgb)
        outputs = self.enc.run(
            [self._out_hr0, self._out_hr1, self._out_embed],
            {self._enc_in[0]: blob},
        )
        return {
            "hr0": outputs[0],
            "hr1": outputs[1],
            "embed": outputs[2],
            "orig_h": h,
            "orig_w": w,
        }

    # -- decoding -----------------------------------------------------------

    def decode(
        self,
        emb: dict,
        points_xy: np.ndarray,
        labels: np.ndarray,
        mask_input: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Predict a mask from point prompts.

        points_xy : (N,2) float, in ORIGINAL image pixel coordinates.
        labels    : (N,)  int, 1 = positive (add), 0 = negative (remove).
        mask_input: optional (1,1,256,256) float low-res logits from a prior call.

        Returns (mask_bool HxW, low_res_logits (1,1,256,256), iou_score).
        """
        h, w = emb["orig_h"], emb["orig_w"]
        scale = np.array([INPUT_SIZE / w, INPUT_SIZE / h], dtype=np.float32)

        coords = (points_xy.astype(np.float32) * scale)[None, ...]  # (1,N,2)
        labs = labels.astype(np.float32)[None, ...]  # (1,N)

        if mask_input is None:
            mask_in = np.zeros((1, 1, LOW_RES, LOW_RES), dtype=np.float32)
            has_mask = np.zeros(1, dtype=np.float32)
        else:
            mask_in = mask_input.astype(np.float32)
            has_mask = np.ones(1, dtype=np.float32)

        feeds = {}
        if self._in_embed:
            feeds[self._in_embed] = emb["embed"]
        if self._in_hr0:
            feeds[self._in_hr0] = emb["hr0"]
        if self._in_hr1:
            feeds[self._in_hr1] = emb["hr1"]
        if self._in_coords:
            feeds[self._in_coords] = coords
        if self._in_labels:
            feeds[self._in_labels] = labs
        if self._in_mask:
            feeds[self._in_mask] = mask_in
        if self._in_has_mask:
            feeds[self._in_has_mask] = has_mask
        if self._in_orig_size:
            feeds[self._in_orig_size] = np.array([h, w], dtype=np.int64)

        # Run every output and classify by shape — export layouts differ:
        # mask logits are 4D (1,M,Hm,Wm); IoU scores are 1D/2D.
        results = self.dec.run(None, feeds)
        named = dict(zip(self._dec_out, results))

        mask_tensors = [a for a in results if a.ndim == 4]
        iou_scores = None
        for name, a in named.items():
            if "iou" in name.lower() and a.ndim <= 2:
                iou_scores = a.flatten()
                break
        if not mask_tensors:
            raise RuntimeError("Decoder returned no 4D mask output")

        # Prefer a full-resolution mask output for display, a 256px one for feedback.
        display = max(mask_tensors, key=lambda a: a.shape[-1])
        feedback = min(mask_tensors, key=lambda a: a.shape[-1])

        # Select the best channel (by IoU if available).
        ch = 0
        if display.shape[1] > 1 and iou_scores is not None and iou_scores.size >= display.shape[1]:
            ch = int(np.argmax(iou_scores[: display.shape[1]]))
        iou = float(iou_scores[ch]) if iou_scores is not None and iou_scores.size > ch else 1.0

        disp_logits = display[0, ch]  # (Hm, Wm)
        fb_logits = feedback[0, min(ch, feedback.shape[1] - 1)]  # (Hf, Wf)

        import cv2

        # Threshold at original frame size.
        if disp_logits.shape != (h, w):
            disp_logits = cv2.resize(disp_logits, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bool = disp_logits > 0.0

        # Normalize feedback to (1,1,256,256) so it can seed the next frame.
        low = cv2.resize(fb_logits, (LOW_RES, LOW_RES), interpolation=cv2.INTER_LINEAR)
        low = low[None, None, ...].astype(np.float32)
        return mask_bool, low, iou
