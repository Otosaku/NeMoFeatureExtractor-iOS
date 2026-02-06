import Accelerate
import Foundation

// MARK: - Pre-emphasis Filter

/// Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
/// - Parameters:
///   - samples: input samples
///   - coeff: pre-emphasis coefficient (typically 0.97)
/// - Returns: filtered samples
func applyPreemphasis(_ samples: [Float], coeff: Float) -> [Float] {
    guard samples.count > 1 else { return samples }

    var output = [Float](repeating: 0, count: samples.count)
    output[0] = samples[0]

    for i in 1..<samples.count {
        output[i] = samples[i] - coeff * samples[i - 1]
    }

    return output
}

// MARK: - Window Functions

/// Генерация symmetric Hann window (как в NeMo с periodic=False)
/// Formula: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
/// - Parameter size: размер окна
/// - Returns: массив значений окна
func createHannWindow(size: Int) -> [Float] {
    guard size > 1 else { return size == 1 ? [1.0] : [] }

    var window = [Float](repeating: 0, count: size)
    let scale = 2.0 * Float.pi / Float(size - 1)

    for i in 0..<size {
        window[i] = 0.5 * (1.0 - cos(scale * Float(i)))
    }

    return window
}

// MARK: - FFT Setup

/// Обёртка над vDSP FFT для переиспользования
final class FFTProcessor {
    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length
    private let nFFT: Int
    private let nFFTBins: Int

    // Pre-allocated buffers (using UnsafeMutableBufferPointer for stable pointers)
    private let realBufferPtr: UnsafeMutableBufferPointer<Float>
    private let imagBufferPtr: UnsafeMutableBufferPointer<Float>

    init(nFFT: Int) {
        self.nFFT = nFFT
        self.nFFTBins = nFFT / 2 + 1
        self.log2n = vDSP_Length(log2(Double(nFFT)))

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }
        self.fftSetup = setup

        // Allocate stable buffers
        let halfN = nFFT / 2
        self.realBufferPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: halfN)
        self.imagBufferPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: halfN)
        realBufferPtr.initialize(repeating: 0)
        imagBufferPtr.initialize(repeating: 0)
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
        realBufferPtr.deallocate()
        imagBufferPtr.deallocate()
    }

    /// Вычисление power spectrum одного фрейма
    /// - Parameters:
    ///   - frame: входной фрейм (уже с применённым окном), размер = nFFT
    ///   - output: выходной буфер для power spectrum, размер = nFFTBins
    func computePowerSpectrum(frame: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>) {
        var splitComplex = DSPSplitComplex(
            realp: realBufferPtr.baseAddress!,
            imagp: imagBufferPtr.baseAddress!
        )

        // Convert to split complex format for real FFT
        // vDSP_ctoz extracts interleaved complex data to split complex
        // For real input stored as float[], stride 2 on DSPComplex means stride 1 on floats
        frame.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { complexPtr in
            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))
        }

        // Perform FFT
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Compute power spectrum: |FFT|^2 = real^2 + imag^2
        // vDSP_fft_zrip returns 2x the FFT, so power is 4x
        // We scale by 1/4 to get the true power spectrum

        // DC component (only real part, no imaginary)
        output[0] = splitComplex.realp[0] * splitComplex.realp[0] / 4.0

        // Nyquist component (packed in imagp[0], only real part)
        output[nFFTBins - 1] = splitComplex.imagp[0] * splitComplex.imagp[0] / 4.0

        // Other bins
        for i in 1..<(nFFTBins - 1) {
            let real = splitComplex.realp[i]
            let imag = splitComplex.imagp[i]
            output[i] = (real * real + imag * imag) / 4.0
        }
    }
}

// MARK: - STFT

/// STFT процессор с pre-allocated буферами
final class STFTProcessor {
    private let config: MelSpectrogramConfig
    private let fftProcessor: FFTProcessor
    private let window: [Float]

    // Pre-allocated buffers
    private var frameBuffer: [Float]
    private var paddedSamples: [Float] = []

    init(config: MelSpectrogramConfig) {
        self.config = config
        self.fftProcessor = FFTProcessor(nFFT: config.nFFT)
        self.window = createHannWindow(size: config.windowSize)
        self.frameBuffer = [Float](repeating: 0, count: config.nFFT)
    }

    /// Вычисление количества фреймов для заданного количества сэмплов
    func frameCount(for sampleCount: Int) -> Int {
        guard sampleCount > 0 else { return 0 }

        if config.center {
            // NeMo formula with center padding:
            // frames = (sampleCount + windowSize) // hopLength
            // This matches NeMo's AudioToMelSpectrogramPreprocessor output
            return (sampleCount + config.windowSize) / config.hopLength
        } else {
            guard sampleCount >= config.windowSize else { return 0 }
            return (sampleCount - config.windowSize) / config.hopLength + 1
        }
    }

    /// Вычисление STFT power spectrum
    /// - Parameters:
    ///   - samples: входные аудио сэмплы
    ///   - output: выходной буфер [nFrames, nFFTBins], row-major
    func computeSTFT(samples: [Float], output: UnsafeMutablePointer<Float>) {
        let nFrames = frameCount(for: samples.count)
        guard nFrames > 0, !samples.isEmpty else { return }

        // Apply center padding if needed
        // PyTorch torch.stft with center=True pads by win_length // 2 on each side
        // pad_mode="constant" means zero padding
        let workingSamples: [Float]
        if config.center {
            let padAmount = config.windowSize / 2  // PyTorch uses win_length // 2
            // Zero padding (pad_mode="constant" in NeMo)
            var padded = [Float](repeating: 0, count: samples.count + config.windowSize)

            // Left padding: zeros (already initialized)
            // Center (original samples)
            for i in 0..<samples.count {
                padded[padAmount + i] = samples[i]
            }
            // Right padding: zeros (already initialized)
            workingSamples = padded
        } else {
            workingSamples = samples
        }

        workingSamples.withUnsafeBufferPointer { samplesPtr in
            for frameIdx in 0..<nFrames {
                let startIdx = frameIdx * config.hopLength

                // Copy frame and apply window
                for i in 0..<config.windowSize {
                    if startIdx + i < workingSamples.count {
                        frameBuffer[i] = samplesPtr[startIdx + i] * window[i]
                    } else {
                        frameBuffer[i] = 0
                    }
                }
                // Zero-pad at the end if windowSize < nFFT
                for i in config.windowSize..<config.nFFT {
                    frameBuffer[i] = 0
                }

                // Compute power spectrum for this frame
                let outputPtr = output.advanced(by: frameIdx * config.nFFTBins)
                frameBuffer.withUnsafeBufferPointer { framePtr in
                    fftProcessor.computePowerSpectrum(frame: framePtr.baseAddress!, output: outputPtr)
                }
            }
        }
    }
}

// MARK: - Math utilities

/// Log transform in-place: output = log(input + epsilon)
func logTransformInPlace(_ data: UnsafeMutablePointer<Float>, count: Int, epsilon: Float) {
    var eps = epsilon
    // Add epsilon
    vDSP_vsadd(data, 1, &eps, data, 1, vDSP_Length(count))
    // Natural log
    var n = Int32(count)
    vvlogf(data, data, &n)
}

/// Per-feature (per-row) normalization: (x - mean) / (std + eps)
/// NeMo computes mean/std using only validCols, but applies to all totalCols
/// - Parameters:
///   - data: 2D data [rows, totalCols], row-major
///   - rows: количество строк (mel bins)
///   - validCols: количество valid столбцов (для mean/std)
///   - totalCols: общее количество столбцов в буфере
///   - epsilon: защита от деления на 0
func normalizePerFeatureInPlace(
    _ data: UnsafeMutablePointer<Float>,
    rows: Int,
    validCols: Int,
    totalCols: Int,
    epsilon: Float
) {
    guard validCols > 1 else { return }  // Need at least 2 samples for std

    for row in 0..<rows {
        let rowPtr = data.advanced(by: row * totalCols)

        // Compute mean using only valid columns
        var mean: Float = 0
        vDSP_meanv(rowPtr, 1, &mean, vDSP_Length(validCols))

        // Subtract mean from ALL columns
        var negMean = -mean
        vDSP_vsadd(rowPtr, 1, &negMean, rowPtr, 1, vDSP_Length(totalCols))

        // Compute std using only valid columns (now data is zero-mean)
        var sumSq: Float = 0
        vDSP_svesq(rowPtr, 1, &sumSq, vDSP_Length(validCols))
        // NeMo uses (N-1) in denominator for std (Bessel's correction)
        let variance = sumSq / Float(validCols - 1)
        let std = sqrtf(variance)

        // Divide ALL columns by (std + epsilon)
        var scale = 1.0 / (std + epsilon)
        vDSP_vsmul(rowPtr, 1, &scale, rowPtr, 1, vDSP_Length(totalCols))
    }
}

/// Simplified version when validCols == totalCols
func normalizePerFeatureInPlace(
    _ data: UnsafeMutablePointer<Float>,
    rows: Int,
    cols: Int,
    epsilon: Float
) {
    normalizePerFeatureInPlace(data, rows: rows, validCols: cols, totalCols: cols, epsilon: epsilon)
}
