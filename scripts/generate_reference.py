#!/usr/bin/env python3
"""
Генерация reference данных для тестирования Swift библиотеки.

Использует реальный препроцессор из NeMo VAD модели.
"""

import json
import numpy as np
import torch
import os
import warnings

warnings.filterwarnings("ignore")

# Параметры
SAMPLE_RATE = 16000


def create_test_audio(duration_sec: float = 0.5) -> np.ndarray:
    """Создание детерминированного тестового аудио."""
    np.random.seed(42)
    n_samples = int(duration_sec * SAMPLE_RATE)
    # Смесь синусоид разных частот
    t = np.arange(n_samples) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +   # A5
        0.1 * np.sin(2 * np.pi * 1760 * t) +  # A6
        0.1 * np.random.randn(n_samples)       # noise
    )
    return audio.astype(np.float32)


def compute_mel_using_model(audio: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Вычисление mel-спектрограммы через препроцессор NeMo модели."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    # Загружаем VAD модель
    model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="vad_multilingual_frame_marblenet"
    )

    # Получаем конфиг препроцессора и конвертируем в dict
    pp_cfg = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)

    # Удаляем _target_ (он не нужен для прямого создания)
    pp_cfg.pop("_target_", None)

    # Меняем нормализацию если нужно
    pp_cfg["normalize"] = "per_feature" if normalize else None

    # Отключаем dither для детерминизма
    pp_cfg["dither"] = 0.0

    # Создаём препроцессор с нужной конфигурацией
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    preprocessor = AudioToMelSpectrogramPreprocessor(**pp_cfg)
    preprocessor.eval()

    # Вычисляем mel
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # [1, samples]
    length_tensor = torch.tensor([len(audio)])

    with torch.no_grad():
        mel, _ = preprocessor(input_signal=audio_tensor, length=length_tensor)

    del model
    return mel.squeeze(0).numpy()  # [n_mels, frames]


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "Tests", "NeMoFeatureExtractorTests", "Resources")
    os.makedirs(output_dir, exist_ok=True)

    # Генерируем тестовое аудио
    audio = create_test_audio(duration_sec=0.5)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio min/max: {audio.min():.4f} / {audio.max():.4f}")

    # Вычисляем mel-спектрограммы
    print("\nComputing mel (no normalization)...")
    mel_no_norm = compute_mel_using_model(audio, normalize=False)

    print("Computing mel (with per-feature normalization)...")
    mel_with_norm = compute_mel_using_model(audio, normalize=True)

    print(f"\nMel (no norm) shape: {mel_no_norm.shape}")
    print(f"Mel (no norm) min/max: {mel_no_norm.min():.4f} / {mel_no_norm.max():.4f}")
    print(f"Mel (with norm) shape: {mel_with_norm.shape}")
    print(f"Mel (with norm) min/max: {mel_with_norm.min():.4f} / {mel_with_norm.max():.4f}")

    # Сохраняем в JSON
    reference_data = {
        "audio": audio.tolist(),
        "sample_rate": SAMPLE_RATE,
        "n_mels": 80,
        "n_fft": 512,
        "hop_length": 160,
        "window_size": 400,  # 25ms * 16000
        "mel_no_normalization": mel_no_norm.tolist(),
        "mel_per_feature_normalization": mel_with_norm.tolist(),
    }

    output_path = os.path.join(output_dir, "reference_data.json")
    with open(output_path, "w") as f:
        json.dump(reference_data, f)

    print(f"\nReference data saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
