#!/usr/bin/env python3
"""
Экспорт mel filterbank из NeMo для использования в Swift.

Это гарантирует точное соответствие с NeMo preprocessing.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import os
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "Resources")
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем модель для получения filterbank
    print("Loading NeMo VAD model...")
    model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="vad_multilingual_frame_marblenet"
    )

    # Получаем preprocessor
    pp_cfg = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
    pp_cfg.pop("_target_", None)

    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    preprocessor = AudioToMelSpectrogramPreprocessor(**pp_cfg)

    # Извлекаем filterbank
    fb = preprocessor.featurizer
    mel_filterbank = fb.fb.cpu().numpy().squeeze(0)  # Remove batch dim: [80, 257]

    print(f"Mel filterbank shape: {mel_filterbank.shape}")
    print(f"Mel filterbank dtype: {mel_filterbank.dtype}")
    print(f"Mel filterbank sum per filter (first 5): {mel_filterbank.sum(axis=1)[:5]}")

    # Параметры
    config = {
        "sample_rate": fb.sample_rate,
        "n_fft": fb.n_fft,
        "hop_length": fb.hop_length,
        "win_length": fb.win_length,
        "n_mels": mel_filterbank.shape[0],
        "n_fft_bins": mel_filterbank.shape[1],
        "log_zero_guard_value": float(fb.log_zero_guard_value),
        "log_zero_guard_type": fb.log_zero_guard_type,
    }

    print(f"\nConfig: {config}")

    # Сохраняем filterbank как JSON
    output_path = os.path.join(output_dir, "nemo_mel_filterbank.json")
    with open(output_path, "w") as f:
        json.dump({
            "config": config,
            "filterbank": mel_filterbank.tolist()
        }, f)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # Также сохраним как бинарный файл (более компактно)
    bin_path = os.path.join(output_dir, "nemo_mel_filterbank.bin")
    mel_filterbank.astype(np.float32).tofile(bin_path)
    print(f"Binary saved to: {bin_path}")
    print(f"Binary size: {os.path.getsize(bin_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
