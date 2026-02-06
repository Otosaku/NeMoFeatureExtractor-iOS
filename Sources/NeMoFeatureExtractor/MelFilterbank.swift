import Accelerate
import Foundation

// MARK: - NeMo Filterbank Loader

enum NeMoFilterbankLoader {
    /// Загрузка pre-computed filterbank из bundle resources
    /// - Returns: filterbank матрица [80, 257], row-major, или nil если файл не найден
    static func loadFromBundle() -> [Float]? {
        // Try with subdirectory first (for .copy), then without (for .process)
        let url = Bundle.module.url(
            forResource: "mel_filterbank",
            withExtension: "bin",
            subdirectory: "Resources"
        ) ?? Bundle.module.url(
            forResource: "mel_filterbank",
            withExtension: "bin"
        )

        guard let url = url else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            // NeMo filterbank: 80 mels x 257 bins = 20560 floats x 4 bytes = 82240 bytes
            let expectedSize = 80 * 257 * MemoryLayout<Float>.size
            guard data.count == expectedSize else {
                print("Warning: Filterbank file size mismatch. Expected \(expectedSize), got \(data.count)")
                return nil
            }

            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }
        } catch {
            print("Warning: Failed to load NeMo filterbank: \(error)")
            return nil
        }
    }
}

// MARK: - Mel Scale Conversion

/// Конвертация частоты в mel scale (HTK formula)
@inline(__always)
func hzToMel(_ hz: Float) -> Float {
    return 2595.0 * log10f(1.0 + hz / 700.0)
}

/// Конвертация mel в частоту (HTK formula)
@inline(__always)
func melToHz(_ mel: Float) -> Float {
    return 700.0 * (powf(10.0, mel / 2595.0) - 1.0)
}

// MARK: - Mel Filterbank

/// Генератор mel filterbank матрицы
enum MelFilterbank {
    /// Создание mel filterbank матрицы
    /// - Parameters:
    ///   - nMels: количество mel-фильтров
    ///   - nFFT: размер FFT
    ///   - sampleRate: sample rate в Hz
    ///   - fMin: минимальная частота
    ///   - fMax: максимальная частота
    ///   - norm: нормализация filterbank
    /// - Returns: filterbank матрица [nMels, nFFTBins], row-major
    static func create(
        nMels: Int,
        nFFT: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float,
        norm: MelNorm = .slaney
    ) -> [Float] {
        let nFFTBins = nFFT / 2 + 1
        let sampleRateF = Float(sampleRate)

        // Mel scale boundaries
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // Create mel points (nMels + 2 points for triangular filters)
        let nPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nPoints)
        let melStep = (melMax - melMin) / Float(nPoints - 1)
        for i in 0..<nPoints {
            melPoints[i] = melMin + Float(i) * melStep
        }

        // Convert mel points to Hz
        let hzPoints = melPoints.map { melToHz($0) }

        // Convert Hz to FFT bin indices (floating point)
        let binPoints = hzPoints.map { hz -> Float in
            return (hz / sampleRateF) * Float(nFFT)
        }

        // Create filterbank matrix [nMels, nFFTBins]
        var filterbank = [Float](repeating: 0, count: nMels * nFFTBins)

        for m in 0..<nMels {
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]

            for k in 0..<nFFTBins {
                let kF = Float(k)

                if kF >= left && kF < center {
                    // Rising slope
                    filterbank[m * nFFTBins + k] = (kF - left) / (center - left)
                } else if kF >= center && kF <= right {
                    // Falling slope
                    filterbank[m * nFFTBins + k] = (right - kF) / (right - center)
                }
                // Otherwise remains 0
            }

            // Apply Slaney normalization if requested
            // Slaney normalization: each filter is normalized so that the area = 1
            // This is done by dividing by the width of the filter in Hz
            if norm == .slaney {
                let filterWidth = hzPoints[m + 2] - hzPoints[m]
                if filterWidth > 0 {
                    let scale = 2.0 / filterWidth
                    for k in 0..<nFFTBins {
                        filterbank[m * nFFTBins + k] *= scale
                    }
                }
            }
        }

        return filterbank
    }

    /// Применение filterbank к power spectrum
    /// - Parameters:
    ///   - filterbank: filterbank матрица [nMels, nFFTBins], row-major
    ///   - powerSpectrum: power spectrum [nFrames, nFFTBins], row-major
    ///   - output: выход [nMels, nFrames], row-major
    ///   - nMels: количество mel-фильтров
    ///   - nFFTBins: количество FFT bins
    ///   - nFrames: количество фреймов
    static func apply(
        filterbank: [Float],
        powerSpectrum: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        nMels: Int,
        nFFTBins: Int,
        nFrames: Int
    ) {
        // Matrix multiplication: filterbank [nMels, nFFTBins] x powerSpectrum^T [nFFTBins, nFrames]
        // Result: [nMels, nFrames]
        //
        // vDSP_mmul: C = A * B
        // A: [M, K], B: [K, N], C: [M, N]
        //
        // Here we have:
        // - filterbank: [nMels, nFFTBins] (M=nMels, K=nFFTBins)
        // - powerSpectrum: [nFrames, nFFTBins] row-major
        //
        // We need powerSpectrum transposed: [nFFTBins, nFrames]
        // So we do: output = filterbank * transpose(powerSpectrum)

        // Transpose powerSpectrum [nFrames, nFFTBins] -> [nFFTBins, nFrames]
        var transposed = [Float](repeating: 0, count: nFFTBins * nFrames)
        vDSP_mtrans(powerSpectrum, 1, &transposed, 1, vDSP_Length(nFFTBins), vDSP_Length(nFrames))

        // Matrix multiply
        filterbank.withUnsafeBufferPointer { filterbankPtr in
            vDSP_mmul(
                filterbankPtr.baseAddress!, 1,  // A: filterbank [nMels, nFFTBins]
                transposed, 1,                   // B: transposed powerSpectrum [nFFTBins, nFrames]
                output, 1,                       // C: output [nMels, nFrames]
                vDSP_Length(nMels),              // M
                vDSP_Length(nFrames),            // N
                vDSP_Length(nFFTBins)            // K
            )
        }
    }
}
