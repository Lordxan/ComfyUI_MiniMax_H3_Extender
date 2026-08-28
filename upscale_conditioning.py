"""Helper to generate upscaled conditioning for the upscaling refinement pass."""

def _duration_to_frames(seconds):
    raw = max(5, int(round(float(seconds) * 24)))
    while raw % 17 != 5:
        raw += 1
    return raw


def _make_upscaled_conditioning(clip, vae, model, original_width, original_height, target_megapixels, clip_prompt, frame_count, ref_items, ref_blocks, active_picture_slots, active_video_slots, selected_audio_slots, audio_native_offset):
    """Generate conditioning at an upscaled resolution.
    
    Returns (upscaled_guider, upscaled_conditioning) or (None, None) if target_megapixels <= 0.
    """
    if target_megapixels <= 0.0:
        return None, None
    
    # Calculate upscaled resolution
    aspect = original_width / float(original_height)
    target_pixels = target_megapixels * 1024 * 1024
    h_raw = (target_pixels / aspect) ** 0.5
    w_raw = h_raw * aspect
    up_h = max(32, round(h_raw / 32) * 32)
    up_w = max(32, round(w_raw / 32) * 32)
    
    # Build conditioning at upscaled resolution
    up_frame_count = _duration_to_frames(frame_count)
    
    # Import from extender
    from .extender import _make_ref2va_conditioning, _BasicGuider
    
    up_positive, up_latent = _make_ref2va_conditioning(
        clip, vae, clip_prompt, up_w, up_h, up_frame_count,
        ref_items, ref_blocks, active_picture_slots, active_video_slots,
        active_audio_slots=selected_audio_slots, audio_native_offset=audio_native_offset,
    )
    
    up_guider = _BasicGuider(model)
    up_guider.set_conds(up_positive)
    
    return up_guider, up_positive
