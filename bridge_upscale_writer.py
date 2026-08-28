"""MiniMax H3 Extender - Bridge Node for Latent Upscaling (Part 2: Cache Writer)

Takes an upscaled video latent and original audio latent, combines them into
a new disk cache file that MiniMaxH3MotionContextDiskFinalDecode can consume.
"""

import json
import os
import time
from pathlib import Path

import torch

from .motion_context_disk import (
    _write_json_atomic,
    _write_tensor_raw,
    _DATA_MAGIC,
    CACHE_VERSION,
    BUILD,
)
from .motion_context_ram import _pixel_frames

CACHE_TYPE = "H3_MOTION_DISK_CACHE"


class H3UpscaleLatentWriter:
    """Write an upscaled latent to a new disk cache file.

    This is the OUTPUT side of the upscaling pipeline. It takes the
    upscaled video latent (from 3D upscaler + refinement pass) and
    the original audio latent (passthrough), then writes them to a
    new .h3cache + .json pair using the same binary format as the Extender.

    The resulting cache handle can be connected directly to
    MiniMaxH3MotionContextDiskFinalDecode for decoding to pixel space
    and muxing into a final MP4.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "original_cache": (CACHE_TYPE,),
            },
        }

    RETURN_TYPES = (CACHE_TYPE,)
    RETURN_NAMES = ("upscaled_cache",)
    FUNCTION = "write"
    CATEGORY = "video/MinimaxH3"

    def write(self, upscaled_video_latent, audio_latent, original_cache):
        # 1. Validate inputs are dicts with required keys
        if not isinstance(upscaled_video_latent, dict) or "samples" not in upscaled_video_latent:
            raise ValueError("H3UpscaleLatentWriter: upscaled_video_latent must be a LATENT dict.")
        if not isinstance(audio_latent, dict) or "samples" not in audio_latent:
            raise ValueError("H3UpscaleLatentWriter: audio_latent must be a LATENT dict.")
        if not isinstance(original_cache, dict):
            raise ValueError("H3UpscaleLatentWriter: original_cache must be a CACHE_TYPE dict.")

        video = upscaled_video_latent["samples"]
        audio = audio_latent["samples"]

        # Ensure tensors on CPU for file writing
        if video.device.type != "cpu":
            video = video.to(device="cpu")
        if audio.device.type != "cpu":
            audio = audio.to(device="cpu")

        # Ensure consistent dimensions (unsqueeze batch dim if needed)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)

        # Validate shapes
        if video.ndim != 5:
            raise ValueError(f"H3UpscaleLatentWriter: video must be 5D [B,C,T,H,W], got {tuple(video.shape)}")
        if audio.ndim != 4:
            raise ValueError(f"H3UpscaleLatentWriter: audio must be 4D [B,C,2,A], got {tuple(audio.shape)}")
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("H3UpscaleLatentWriter: batch size must be 1.")
        if video.shape[1] != 24:
            raise ValueError(f"H3UpscaleLatentWriter: expected 24 video channels, got {video.shape[1]}")
        if audio.shape[1] != 32:
            raise ValueError(f"H3UpscaleLatentWriter: expected 32 audio channels, got {audio.shape[1]}")

        # 2. Load original manifest metadata for inheritance
        data_path = Path(original_cache["data_path"])
        manifest_path = Path(original_cache["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", -1)) != CACHE_VERSION:
            raise ValueError(f"H3UpscaleLatentWriter: cache version {manifest.get('version')} incompatible")

        # 3. Construct new file paths (append _upscaled suffix)
        original_stem = Path(data_path.stem)
        new_stem = str(original_stem) + "_upscaled"
        new_data_path = data_path.with_name(f"{new_stem}.h3cache")
        new_manifest_path = manifest_path.with_name(f"{new_stem}.json")
        new_data_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Write binary data file
        with open(new_data_path, "wb") as f:
            f.write(_DATA_MAGIC)
            f.flush()
            os.fsync(f.fileno())

            video_spec = _write_tensor_raw(f, video)
            audio_spec = _write_tensor_raw(f, audio)
            segment_end = int(f.tell())
            f.flush()
            os.fsync(f.fileno())

        # 5. Build new manifest
        new_manifest = dict(manifest)
        new_manifest["version"] = CACHE_VERSION
        new_manifest["build"] = BUILD + "_upscaled"
        new_manifest["updated_at"] = time.time()

        # Update final_frame_count: convert latent T to pixel frame count
        latent_t = int(video.shape[2])
        new_manifest["final_frame_count"] = _pixel_frames(latent_t)

        # Update geometry (latent-space dimensions)
        new_geometry = dict(manifest.get("geometry", {}))
        new_geometry["video_w"] = int(video.shape[4])   # latent W'
        new_geometry["video_h"] = int(video.shape[3])   # latent H'
        new_geometry["video_batch"] = int(video.shape[0])
        new_geometry["video_channels"] = int(video.shape[1])
        new_geometry["audio_channels"] = int(audio.shape[1])
        new_geometry["audio_batch"] = int(audio.shape[0])
        new_geometry["audio_planes"] = int(audio.shape[2])
        new_manifest["geometry"] = new_geometry

        # Create single segment entry
        new_segment = {
            "index": 0,
            "validated": False,
            "frames": _pixel_frames(latent_t),
            "trim_frames": 0,
            "video": video_spec,
            "audio": audio_spec,
            "segment_end": segment_end,
        }
        new_manifest["segments"] = [new_segment]

        # 6. Write manifest atomically
        _write_json_atomic(new_manifest_path, new_manifest)

        # 7. Create and return cache handle
        cache_handle = {
            "version": CACHE_VERSION,
            "data_path": str(new_data_path.resolve()),
            "manifest_path": str(new_manifest_path.resolve()),
            "run_mode": str(original_cache.get("run_mode", "full_batch")),
            "stop": False,
            "next_index": 0,
            "status": f"upscaled from: {str(original_cache.get('status', ''))}",
        }

        return (cache_handle,)


NODE_CLASS_MAPPINGS = {"H3UpscaleLatentWriter": H3UpscaleLatentWriter}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UpscaleLatentWriter": "MiniMax H3 Upscale Cache Writer"}
