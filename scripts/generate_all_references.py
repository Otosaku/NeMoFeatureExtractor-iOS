#!/usr/bin/env python3
"""
Generate reference features from original NeMo models WITHOUT any modifications.

Models:
1. vad_multilingual_frame_marblenet (VAD)
2. speakerverification_en_titanet_small (Speaker)
3. nvidia/parakeet-tdt_ctc-110m (ASR)
"""

import json
import numpy as np
import torch
import os
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


def create_test_audio(duration_sec: float = 0.5) -> np.ndarray:
    """Create deterministic test audio."""
    np.random.seed(42)
    n_samples = int(duration_sec * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.sin(2 * np.pi * 1760 * t) +
        0.1 * np.random.randn(n_samples)
    )
    return audio.astype(np.float32)


def get_preprocessor_config(model):
    """Extract preprocessor config from model."""
    from omegaconf import OmegaConf
    pp_cfg = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
    pp_cfg.pop("_target_", None)
    return pp_cfg


def compute_features(audio: np.ndarray, preprocessor, dither: float = 0.0):
    """Compute features using preprocessor."""
    # Temporarily disable dither for determinism
    original_dither = preprocessor.featurizer.dither
    preprocessor.featurizer.dither = dither
    preprocessor.eval()

    audio_tensor = torch.tensor(audio).unsqueeze(0)
    length_tensor = torch.tensor([len(audio)])

    with torch.no_grad():
        features, _ = preprocessor(input_signal=audio_tensor, length=length_tensor)

    # Restore original dither
    preprocessor.featurizer.dither = original_dither

    return features.squeeze(0).numpy()


def main():
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    output_dir = os.path.join(os.path.dirname(__file__), "..", "Tests", "NeMoFeatureExtractorTests", "Resources")
    os.makedirs(output_dir, exist_ok=True)

    # Create test audio
    audio = create_test_audio(duration_sec=0.5)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio min/max: {audio.min():.4f} / {audio.max():.4f}")
    print()

    results = {
        "audio": audio.tolist(),
        "sample_rate": SAMPLE_RATE,
        "models": {}
    }

    # ========== 1. VAD Model ==========
    print("=" * 50)
    print("1. Loading VAD model: vad_multilingual_frame_marblenet")
    print("=" * 50)

    vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="vad_multilingual_frame_marblenet"
    )

    vad_cfg = get_preprocessor_config(vad_model)
    print(f"VAD preprocessor config:")
    for k, v in vad_cfg.items():
        print(f"  {k}: {v}")

    vad_features = compute_features(audio, vad_model.preprocessor)
    print(f"VAD features shape: {vad_features.shape}")
    print(f"VAD features min/max: {vad_features.min():.4f} / {vad_features.max():.4f}")
    print()

    results["models"]["vad_marblenet"] = {
        "model_name": "vad_multilingual_frame_marblenet",
        "config": vad_cfg,
        "features": vad_features.tolist(),
        "shape": list(vad_features.shape)
    }

    del vad_model

    # ========== 2. Speaker Model ==========
    print("=" * 50)
    print("2. Loading Speaker model: titanet_small")
    print("=" * 50)

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_small"
    )
    speaker_cfg = get_preprocessor_config(speaker_model)
    print(f"Speaker preprocessor config:")
    for k, v in speaker_cfg.items():
        print(f"  {k}: {v}")

    speaker_features = compute_features(audio, speaker_model.preprocessor)
    print(f"Speaker features shape: {speaker_features.shape}")
    print(f"Speaker features min/max: {speaker_features.min():.4f} / {speaker_features.max():.4f}")
    print()

    results["models"]["speaker_titanet"] = {
        "model_name": "titanet_small",
        "config": speaker_cfg,
        "features": speaker_features.tolist(),
        "shape": list(speaker_features.shape)
    }

    del speaker_model

    # ========== 3. ASR Model ==========
    print("=" * 50)
    print("3. Loading ASR model: nvidia/parakeet-tdt_ctc-110m")
    print("=" * 50)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt_ctc-110m"
    )
    asr_cfg = get_preprocessor_config(asr_model)
    print(f"ASR preprocessor config:")
    for k, v in asr_cfg.items():
        print(f"  {k}: {v}")

    asr_features = compute_features(audio, asr_model.preprocessor)
    print(f"ASR features shape: {asr_features.shape}")
    print(f"ASR features min/max: {asr_features.min():.4f} / {asr_features.max():.4f}")
    print()

    results["models"]["asr_parakeet"] = {
        "model_name": "nvidia/parakeet-tdt_ctc-110m",
        "config": asr_cfg,
        "features": asr_features.tolist(),
        "shape": list(asr_features.shape)
    }

    del asr_model

    # ========== Save Results ==========
    output_path = os.path.join(output_dir, "reference_data.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

    print("=" * 50)
    print(f"Reference data saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("=" * 50)


if __name__ == "__main__":
    main()
