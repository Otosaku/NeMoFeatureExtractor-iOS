import Foundation

/// Режим нормализации mel-спектрограммы
public enum NormalizationMode: Sendable {
    /// Без нормализации (для VAD/MarbleNet)
    case none
    /// Per-feature нормализация: (x - mean) / std для каждого mel-бэнда (для ASR/TitaNet)
    case perFeature
}

/// Режим нормализации mel filterbank
public enum MelNorm: Sendable {
    /// Без нормализации area
    case none
    /// Slaney normalization (area = 1 для каждого фильтра)
    case slaney
}

/// Конфигурация для извлечения mel-спектрограмм
public struct MelSpectrogramConfig: Sendable {
    /// Sample rate в Hz
    public let sampleRate: Int
    /// Количество mel-фильтров
    public let nMels: Int
    /// Размер FFT (для zero-padding)
    public let nFFT: Int
    /// Размер окна в сэмплах (может быть меньше nFFT)
    public let windowSize: Int
    /// Шаг окна в сэмплах
    public let hopLength: Int
    /// Нижняя граница частоты для mel-фильтров (Hz)
    public let fMin: Float
    /// Верхняя граница частоты для mel-фильтров (Hz, nil = sampleRate/2)
    public let fMax: Float?
    /// Режим нормализации спектрограммы
    public let normalization: NormalizationMode
    /// Нормализация mel filterbank
    public let melNorm: MelNorm
    /// Epsilon для log transform
    public let logEpsilon: Float
    /// Epsilon для нормализации (защита от деления на 0)
    public let normEpsilon: Float
    /// Center padding for STFT (как в librosa/NeMo)
    public let center: Bool
    /// Pre-emphasis coefficient (NeMo default: 0.97). Set to nil to disable.
    public let preemph: Float?
    /// Pad output to multiple of this value (VAD=2, Speaker=16, ASR=0)
    public let padTo: Int

    public init(
        sampleRate: Int = 16000,
        nMels: Int = 80,
        nFFT: Int = 512,
        windowSize: Int = 400,
        hopLength: Int = 160,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        normalization: NormalizationMode = .none,
        melNorm: MelNorm = .slaney,
        logEpsilon: Float = 1e-5,
        normEpsilon: Float = 1e-5,
        center: Bool = true,
        preemph: Float? = 0.97,
        padTo: Int = 0
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.windowSize = windowSize
        self.hopLength = hopLength
        self.fMin = fMin
        self.fMax = fMax
        self.normalization = normalization
        self.melNorm = melNorm
        self.logEpsilon = logEpsilon
        self.normEpsilon = normEpsilon
        self.center = center
        self.preemph = preemph
        self.padTo = padTo
    }

    /// Количество FFT bins (nFFT / 2 + 1)
    public var nFFTBins: Int { nFFT / 2 + 1 }

    /// Эффективная верхняя граница частоты
    public var effectiveFMax: Float { fMax ?? Float(sampleRate) / 2.0 }
}

// MARK: - Presets

extension MelSpectrogramConfig {
    /// NeMo log_zero_guard_value = 2^(-24)
    private static let nemoLogEpsilon: Float = 5.960464477539063e-08

    /// Конфигурация для NeMo VAD (MarbleNet) - без нормализации
    public static let nemoVAD = MelSpectrogramConfig(
        sampleRate: 16000,
        nMels: 80,
        nFFT: 512,
        windowSize: 400,  // 25ms at 16kHz
        hopLength: 160,   // 10ms at 16kHz
        fMin: 0.0,
        fMax: nil,
        normalization: .none,
        melNorm: .slaney,
        logEpsilon: nemoLogEpsilon,
        center: true,
        preemph: 0.97,
        padTo: 2
    )

    /// Конфигурация для NeMo ASR (Parakeet и др.) - с per-feature нормализацией
    public static let nemoASR = MelSpectrogramConfig(
        sampleRate: 16000,
        nMels: 80,
        nFFT: 512,
        windowSize: 400,
        hopLength: 160,
        fMin: 0.0,
        fMax: nil,
        normalization: .perFeature,
        melNorm: .slaney,
        logEpsilon: nemoLogEpsilon,
        center: true,
        preemph: 0.97,
        padTo: 0
    )

    /// Конфигурация для NeMo Speaker (TitaNet и др.) - с per-feature нормализацией
    public static let nemoSpeaker = MelSpectrogramConfig(
        sampleRate: 16000,
        nMels: 80,
        nFFT: 512,
        windowSize: 400,
        hopLength: 160,
        fMin: 0.0,
        fMax: nil,
        normalization: .perFeature,
        melNorm: .slaney,
        logEpsilon: nemoLogEpsilon,
        center: true,
        preemph: 0.97,
        padTo: 16
    )
}
