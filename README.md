# NeMoFeatureExtractor

A Swift library for extracting mel-spectrogram features compatible with NVIDIA NeMo speech models. Designed for iOS/macOS applications using CoreML.

## Features

- Exact compatibility with NeMo's feature extraction pipeline
- Supports VAD, Speaker Recognition, and ASR models
- High performance using Apple's Accelerate framework (vDSP)
- Pre-computed mel filterbank from NeMo for maximum accuracy
- Output as `[[Float]]` or `MLMultiArray` for CoreML inference

## Supported Models

| Model Type | Config Preset | Use Case |
|------------|---------------|----------|
| VAD | `.nemoVAD` | Voice Activity Detection (MarbleNet) |
| Speaker | `.nemoSpeaker` | Speaker Verification/Identification (TitaNet) |
| ASR | `.nemoASR` | Speech Recognition (Parakeet, Conformer) |

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/niceguy135/NeMoFeatureExtractor-iOS.git", from: "1.0.0")
]
```

Or in Xcode: File → Add Package Dependencies → Enter repository URL.

## Usage

### Basic Usage

```swift
import NeMoFeatureExtractor

// Create extractor with desired config
let extractor = NeMoFeatureExtractor(config: .nemoVAD)

// Process audio samples (Float32, mono, 16kHz)
let audioSamples: [Float] = loadAudio() // Your audio loading code
let features = try extractor.process(samples: audioSamples)
// features: [[Float]] with shape [80, numFrames]
```

### For CoreML Inference

```swift
let extractor = NeMoFeatureExtractor(config: .nemoSpeaker)

// Get MLMultiArray directly for CoreML
let mlFeatures = try extractor.processToMLMultiArray(samples: audioSamples)
// mlFeatures: MLMultiArray with shape [1, 80, numFrames]

// Use with your CoreML model
let prediction = try model.prediction(audio_signal: mlFeatures)
```

### Custom Configuration

```swift
let customConfig = MelSpectrogramConfig(
    sampleRate: 16000,
    nMels: 80,
    nFFT: 512,
    windowSize: 400,    // 25ms
    hopLength: 160,     // 10ms
    fMin: 0.0,
    fMax: nil,          // Nyquist frequency
    normalization: .perFeature,
    melNorm: .slaney,
    logEpsilon: 5.960464477539063e-08,  // 2^-24
    center: true,
    preemph: 0.97,
    padTo: 16
)

let extractor = NeMoFeatureExtractor(config: customConfig)
```

## Configuration Presets

### VAD (`.nemoVAD`)
- No normalization
- `padTo: 2`
- For MarbleNet and similar VAD models

### Speaker (`.nemoSpeaker`)
- Per-feature normalization
- `padTo: 16`
- For TitaNet and speaker embedding models

### ASR (`.nemoASR`)
- Per-feature normalization
- `padTo: 0` (no padding)
- For Parakeet, Conformer, and other ASR models

## Technical Details

### Processing Pipeline

1. **Pre-emphasis**: `y[n] = x[n] - 0.97 * x[n-1]`
2. **STFT**: Center-padded, Hann window (symmetric)
3. **Power Spectrum**: `|FFT|²`
4. **Mel Filterbank**: 80 mel bands, Slaney normalization
5. **Log Transform**: `log(mel + epsilon)`
6. **Normalization**: Per-feature mean/std (optional)
7. **Padding**: To multiple of `padTo` (optional)

### Accuracy

Tested against NeMo Python reference with maximum difference < 6e-05 (floating point precision).

## Requirements

- iOS 14.0+ / macOS 11.0+
- Swift 5.9+

## License

MIT License

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Original Python implementation
- Apple Accelerate framework for optimized DSP operations
