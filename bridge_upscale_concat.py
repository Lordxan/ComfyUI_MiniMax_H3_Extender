"""MiniMax H3 Extender - Bridge Node for Latent Upscaling Concatenation

Reads the raw low-resolution latents from the Extender disk cache (all segments)
and concatenates them into a single combined latent. This combined latent can then
be passed to MinimaxH3LatentUpscaler3D for upscaling as one continuous sequence.

This solves the issue where only the last clip was visible during upscaling:
instead of upscaling one clip at a time, all clips are concatenated first,
then upscaler'd together, preserving temporal continuity across clip boundaries.

Workflow example:
  [Extender] → cache → [H3UpscaleConcat] → concatenated_latent →
               [MinimaxH3LatentUpscaler3D] → upscaled → [Sampler] → [Writer] → Preview
"""

import torch
import comfy.nested_tensor
from pathlib import Path

from .motion_context_disk import (
    _load_manifest_from_paths,
    _load_segment_video,
    _load_segment_audio,
)

CACHE_TYPE = "H3_MOTION_DISK_CACHE"


class H3UpscaleConcat:
    """Concatenate all low-resolution segments from the Extender cache.

    This node reads the raw Extender cache, extracts video and audio tensors
    from every segment, and concatenates them into a single combined latent.
    The combined latent preserves the original resolution and can be fed into
    the upscaler chain for processing as one continuous sequence.

    This is the INPUT side of the upscaling pipeline. The actual upscaling
    is performed by MinimaxH3LatentUpscaler3D (from the Comfyui_Minimax_h3_latent_Upscaler
    package), not by this node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": (CACHE_TYPE,),
            },
        }

    RETURN_TYPES = (
        "LATENT",      # concatenated_latent - feeds into upscaler
        CACHE_TYPE,    # original_cache - passes through for writer
    )
    RETURN_NAMES = ("concatenated_latent", "original_cache")
    FUNCTION = "concat"
    CATEGORY = "video/MinimaxH3"

    def concat(self, cache):
        # Validate cache is a dict with required keys
        if not isinstance(cache, dict):
            raise ValueError(f"H3UpscaleConcat: expected CACHE_TYPE dict, got {type(cache)}")
        if "data_path" not in cache or "manifest_path" not in cache:
            raise ValueError("H3UpscaleConcat: cache missing data_path or manifest_path")

        data_path = Path(cache["data_path"])
        manifest_path = Path(cache["manifest_path"])

        # Load manifest
        manifest = _load_manifest_from_paths(data_path, manifest_path)
        if manifest is None:
            raise ValueError(f"H3UpscaleConcat: manifest not found at {manifest_path}")

        segments = manifest.get("segments", [])
        if not segments:
            raise ValueError("H3UpscaleConcat: cache contains no segments/clips.")

        # Extract video and audio tensors from each segment
        video_tensors = []
        audio_tensors = []

        for seg in segments:
            video = _load_segment_video(data_path, seg)   # [B, C, T, H, W]
            audio = _load_segment_audio(data_path, seg)    # [B, C, 2, A]
            video_tensors.append(video)
            audio_tensors.append(audio)

        # Concatenate along temporal dimension (T for video, A for audio)
        combined_video = torch.cat(video_tensors, dim=2)    # [B, C, T_total, H, W]
        combined_audio = torch.cat(audio_tensors, dim=3)    # [B, C, 2, A_total]

        # Build joint latent with NestedTensor
        joint = comfy.nested_tensor.NestedTensor((combined_video, combined_audio))

        # Return as standard ComfyUI LATENT dict for upscaler input
        return (
            {"samples": joint},      # concatenated_latent -> feeds upscaler
            cache,                    # original_cache -> passes through
        )


NODE_CLASS_MAPPINGS = {"H3UpscaleConcat": H3UpscaleConcat}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UpscaleConcat": "MiniMax H3 Upscale Concat"}