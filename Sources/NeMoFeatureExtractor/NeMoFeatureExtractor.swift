import Accelerate
import CoreML
import Foundation

/// Ошибки NeMoFeatureExtractor
public enum NeMoFeatureExtractorError: Error, LocalizedError {
    case invalidInput(String)
    case coreMLConversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .coreMLConversionFailed(let message):
            return "CoreML conversion failed: \(message)"
        }
    }
}

/// Извлекатель mel-спектрограмм, совместимый с NeMo моделями
///
/// Поддерживает:
/// - VAD (MarbleNet) - без нормализации
/// - ASR (FastConformer и др.) - с per-feature нормализацией
/// - Speaker (TitaNet и др.) - с per-feature нормализацией
///
/// Пример использования:
/// ```swift
/// // Для VAD
/// let vadExtractor = NeMoFeatureExtractor(config: .nemoVAD)
/// let features = try vadExtractor.process(samples: audioSamples)
///
/// // Для ASR/TitaNet
/// let asrExtractor = NeMoFeatureExtractor(config: .nemoASR)
/// let features = try asrExtractor.process(samples: audioSamples)
/// ```
public final class NeMoFeatureExtractor {
    /// Конфигурация
    public let config: MelSpectrogramConfig

    // Internal processors
    private let stftProcessor: STFTProcessor
    private let filterbank: [Float]

    // Pre-allocated buffers (resized as needed)
    private var powerSpectrumBuffer: [Float] = []
    private var melBuffer: [Float] = []

    /// Инициализация с конфигурацией
    /// - Parameters:
    ///   - config: конфигурация mel-спектрограммы
    ///   - useNeMoFilterbank: использовать pre-computed filterbank из NeMo (рекомендуется для точности)
    public init(config: MelSpectrogramConfig = .nemoVAD, useNeMoFilterbank: Bool = true) {
        self.config = config
        self.stftProcessor = STFTProcessor(config: config)

        // Пытаемся загрузить pre-computed filterbank из NeMo для максимальной точности
        if useNeMoFilterbank,
           config.nMels == 80,
           config.nFFTBins == 257,
           let nemoFilterbank = NeMoFilterbankLoader.loadFromBundle() {
            self.filterbank = nemoFilterbank
        } else {
            // Fallback: генерируем filterbank
            self.filterbank = MelFilterbank.create(
                nMels: config.nMels,
                nFFT: config.nFFT,
                sampleRate: config.sampleRate,
                fMin: config.fMin,
                fMax: config.effectiveFMax,
                norm: config.melNorm
            )
        }
    }

    // MARK: - Public API

    /// Вычисление mel-спектрограммы из аудио сэмплов
    /// - Parameter samples: аудио сэмплы (Float32, mono, 16kHz)
    /// - Returns: mel-спектрограмма как 2D массив [nMels, nFrames]
    public func process(samples: [Float]) throws -> [[Float]] {
        // Step 0: Apply pre-emphasis if configured
        let processedSamples: [Float]
        if let preemph = config.preemph {
            processedSamples = applyPreemphasis(samples, coeff: preemph)
        } else {
            processedSamples = samples
        }

        let nFrames = stftProcessor.frameCount(for: processedSamples.count)
        guard nFrames > 0 else {
            throw NeMoFeatureExtractorError.invalidInput(
                "Audio too short: \(samples.count) samples, need at least \(config.windowSize)"
            )
        }

        // NeMo's valid frame count (for normalization and masking): audio_length // hop_length
        // This is less than nFrames from STFT due to center padding
        let validFrames = samples.count / config.hopLength

        // Calculate padded frame count based on nFrames (STFT output), not validFrames
        // NeMo outputs all STFT frames, padded to multiple of padTo
        let paddedFrames = computePaddedFrameCount(nFrames)

        // Ensure buffers are large enough
        ensureBufferCapacity(nFrames: max(nFrames, paddedFrames))

        // Step 1: STFT -> power spectrum [nFrames, nFFTBins]
        stftProcessor.computeSTFT(samples: processedSamples, output: &powerSpectrumBuffer)

        // Step 2: Apply mel filterbank -> [nMels, nFrames]
        MelFilterbank.apply(
            filterbank: filterbank,
            powerSpectrum: powerSpectrumBuffer,
            output: &melBuffer,
            nMels: config.nMels,
            nFFTBins: config.nFFTBins,
            nFrames: nFrames
        )

        // Step 3: Log transform
        logTransformInPlace(&melBuffer, count: config.nMels * nFrames, epsilon: config.logEpsilon)

        // Step 4: Normalize if needed (NeMo uses only validFrames for mean/std)
        // Buffer layout is [nMels, nFrames], so totalCols = nFrames
        if config.normalization == .perFeature {
            normalizePerFeatureInPlace(
                &melBuffer,
                rows: config.nMels,
                validCols: validFrames,  // Only use valid frames for mean/std calculation
                totalCols: nFrames,       // But buffer has nFrames columns per row
                epsilon: config.normEpsilon
            )
        }

        // Step 5: Mask frames beyond validFrames (NeMo sets to 0)
        maskFramesBeyond(validFrames: validFrames, totalFrames: nFrames)

        // Step 6: Apply padding if needed (based on nFrames, not validFrames)
        // NeMo outputs STFT frame count, padded to multiple of padTo
        var finalFrames = nFrames
        if config.padTo > 0 {
            let padAmount = nFrames % config.padTo
            if padAmount != 0 {
                let padFrames = config.padTo - padAmount
                finalFrames = nFrames + padFrames
            }
        }

        // Guard against zero frames
        guard finalFrames > 0 else {
            throw NeMoFeatureExtractorError.invalidInput(
                "Audio too short: \\(samples.count) samples produces 0 valid frames"
            )
        }

        // Repack buffer if needed (melBuffer has nFrames columns, we need finalFrames)
        // This handles both padding and stride mismatch
        if nFrames != finalFrames {
            repackMelBuffer(fromCols: nFrames, toCols: finalFrames, validCols: validFrames)
        }

        // Convert to 2D array [nMels, finalFrames]
        return stride(from: 0, to: config.nMels * finalFrames, by: finalFrames).map { start in
            Array(melBuffer[start..<(start + finalFrames)])
        }
    }

    /// Вычисление mel-спектрограммы и возврат как MLMultiArray
    /// - Parameter samples: аудио сэмплы (Float32, mono, 16kHz)
    /// - Returns: MLMultiArray с shape [1, nMels, nFrames]
    public func processToMLMultiArray(samples: [Float]) throws -> MLMultiArray {
        // Step 0: Apply pre-emphasis if configured
        let processedSamples: [Float]
        if let preemph = config.preemph {
            processedSamples = applyPreemphasis(samples, coeff: preemph)
        } else {
            processedSamples = samples
        }

        let nFrames = stftProcessor.frameCount(for: processedSamples.count)
        guard nFrames > 0 else {
            throw NeMoFeatureExtractorError.invalidInput(
                "Audio too short: \(samples.count) samples, need at least \(config.windowSize)"
            )
        }

        // NeMo's valid frame count (for normalization and masking): audio_length // hop_length
        let validFrames = samples.count / config.hopLength

        // Calculate padded frame count based on nFrames (STFT output), not validFrames
        let paddedFrames = computePaddedFrameCount(nFrames)

        // Ensure buffers are large enough
        ensureBufferCapacity(nFrames: max(nFrames, paddedFrames))

        // Step 1: STFT -> power spectrum [nFrames, nFFTBins]
        stftProcessor.computeSTFT(samples: processedSamples, output: &powerSpectrumBuffer)

        // Step 2: Apply mel filterbank -> [nMels, nFrames]
        MelFilterbank.apply(
            filterbank: filterbank,
            powerSpectrum: powerSpectrumBuffer,
            output: &melBuffer,
            nMels: config.nMels,
            nFFTBins: config.nFFTBins,
            nFrames: nFrames
        )

        // Step 3: Log transform
        logTransformInPlace(&melBuffer, count: config.nMels * nFrames, epsilon: config.logEpsilon)

        // Step 4: Normalize if needed (NeMo uses only validFrames for mean/std)
        // Buffer layout is [nMels, nFrames], so totalCols = nFrames
        if config.normalization == .perFeature {
            normalizePerFeatureInPlace(
                &melBuffer,
                rows: config.nMels,
                validCols: validFrames,  // Only use valid frames for mean/std calculation
                totalCols: nFrames,       // But buffer has nFrames columns per row
                epsilon: config.normEpsilon
            )
        }

        // Step 5: Mask frames beyond validFrames
        maskFramesBeyond(validFrames: validFrames, totalFrames: nFrames)

        // Step 6: Apply padding if needed (based on nFrames, not validFrames)
        // NeMo outputs STFT frame count, padded to multiple of padTo
        var finalFrames = nFrames
        if config.padTo > 0 {
            let padAmount = nFrames % config.padTo
            if padAmount != 0 {
                let padFrames = config.padTo - padAmount
                finalFrames = nFrames + padFrames
            }
        }

        // Guard against zero frames
        guard finalFrames > 0 else {
            throw NeMoFeatureExtractorError.invalidInput(
                "Audio too short: \\(samples.count) samples produces 0 valid frames"
            )
        }

        // Repack buffer if needed (melBuffer has nFrames columns, we need finalFrames)
        if nFrames != finalFrames {
            repackMelBuffer(fromCols: nFrames, toCols: finalFrames, validCols: validFrames)
        }

        // Create MLMultiArray [1, nMels, finalFrames]
        return try createMLMultiArray(nFrames: finalFrames)
    }

    /// Вычисление количества выходных фреймов для заданного количества сэмплов
    /// - Parameter sampleCount: количество входных сэмплов
    /// - Returns: количество выходных фреймов
    public func outputFrameCount(for sampleCount: Int) -> Int {
        return stftProcessor.frameCount(for: sampleCount)
    }

    /// Вычисление необходимого количества сэмплов для заданной длительности
    /// - Parameter seconds: длительность в секундах
    /// - Returns: количество сэмплов
    public func sampleCount(forDuration seconds: Double) -> Int {
        return Int(seconds * Double(config.sampleRate))
    }

    // MARK: - Private

    private func computePaddedFrameCount(_ nFrames: Int) -> Int {
        guard config.padTo > 0 else { return nFrames }
        let padAmount = nFrames % config.padTo
        if padAmount == 0 { return nFrames }
        return nFrames + (config.padTo - padAmount)
    }

    private func ensureBufferCapacity(nFrames: Int) {
        let powerSpectrumSize = nFrames * config.nFFTBins
        let melSize = config.nMels * nFrames

        if powerSpectrumBuffer.count < powerSpectrumSize {
            powerSpectrumBuffer = [Float](repeating: 0, count: powerSpectrumSize)
        }
        if melBuffer.count < melSize {
            melBuffer = [Float](repeating: 0, count: melSize)
        }
    }

    /// Mask frames beyond validFrames (set to 0, like NeMo's masked_fill)
    private func maskFramesBeyond(validFrames: Int, totalFrames: Int) {
        guard validFrames < totalFrames else { return }

        let nMels = config.nMels
        for mel in 0..<nMels {
            for frame in validFrames..<totalFrames {
                melBuffer[mel * totalFrames + frame] = 0
            }
        }
    }

    /// Repack melBuffer from one column count to another
    /// - Parameters:
    ///   - fromCols: current number of columns in melBuffer
    ///   - toCols: target number of columns
    ///   - validCols: number of valid columns to copy (rest will be zeros)
    private func repackMelBuffer(fromCols: Int, toCols: Int, validCols: Int) {
        let nMels = config.nMels
        let copyCount = min(validCols, toCols)
        var newBuffer = [Float](repeating: 0, count: nMels * toCols)

        for mel in 0..<nMels {
            let srcStart = mel * fromCols
            let dstStart = mel * toCols
            for frame in 0..<copyCount {
                newBuffer[dstStart + frame] = melBuffer[srcStart + frame]
            }
            // Remaining frames are already 0 (padding)
        }

        melBuffer = newBuffer
    }

    private func createMLMultiArray(nFrames: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: config.nMels), NSNumber(value: nFrames)]

        do {
            let mlArray = try MLMultiArray(shape: shape, dataType: .float32)

            // Copy data - melBuffer is [nMels, nFrames] row-major
            // MLMultiArray expects the same layout
            let ptr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: config.nMels * nFrames)
            melBuffer.withUnsafeBufferPointer { srcPtr in
                ptr.update(from: srcPtr.baseAddress!, count: config.nMels * nFrames)
            }

            return mlArray
        } catch {
            throw NeMoFeatureExtractorError.coreMLConversionFailed(error.localizedDescription)
        }
    }
}

// MARK: - Convenience Extensions

extension NeMoFeatureExtractor {
    /// Обработка Double сэмплов (конвертация в Float)
    public func process(samples: [Double]) throws -> [[Float]] {
        let floatSamples = samples.map { Float($0) }
        return try process(samples: floatSamples)
    }

    /// Обработка Double сэмплов в MLMultiArray
    public func processToMLMultiArray(samples: [Double]) throws -> MLMultiArray {
        let floatSamples = samples.map { Float($0) }
        return try processToMLMultiArray(samples: floatSamples)
    }
}
