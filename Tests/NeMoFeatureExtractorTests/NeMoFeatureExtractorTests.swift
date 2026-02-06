import XCTest
@testable import NeMoFeatureExtractor

final class NeMoFeatureExtractorTests: XCTestCase {

    // MARK: - Reference Data

    struct ModelReference: Codable {
        let modelName: String
        let config: [String: AnyCodable]
        let features: [[Float]]
        let shape: [Int]

        enum CodingKeys: String, CodingKey {
            case modelName = "model_name"
            case config
            case features
            case shape
        }
    }

    struct ReferenceData: Codable {
        let audio: [Float]
        let sampleRate: Int
        let models: [String: ModelReference]

        enum CodingKeys: String, CodingKey {
            case audio
            case sampleRate = "sample_rate"
            case models
        }
    }

    // Helper for decoding dynamic config values
    struct AnyCodable: Codable {
        let value: Any

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let intVal = try? container.decode(Int.self) {
                value = intVal
            } else if let doubleVal = try? container.decode(Double.self) {
                value = doubleVal
            } else if let boolVal = try? container.decode(Bool.self) {
                value = boolVal
            } else if let stringVal = try? container.decode(String.self) {
                value = stringVal
            } else if container.decodeNil() {
                value = NSNull()
            } else {
                value = ""
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if let val = value as? Int { try container.encode(val) }
            else if let val = value as? Double { try container.encode(val) }
            else if let val = value as? Bool { try container.encode(val) }
            else if let val = value as? String { try container.encode(val) }
            else { try container.encodeNil() }
        }
    }

    var referenceData: ReferenceData!

    override func setUpWithError() throws {
        let bundle = Bundle.module
        guard let url = bundle.url(forResource: "reference_data", withExtension: "json", subdirectory: "Resources") else {
            XCTFail("Reference data file not found")
            return
        }

        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        referenceData = try decoder.decode(ReferenceData.self, from: data)
    }

    // MARK: - Basic Tests

    func testBasicExtraction() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)

        let sampleRate = 16000
        let duration = 0.5
        let samples = (0..<Int(Double(sampleRate) * duration)).map { i in
            Float(sin(2.0 * .pi * 440.0 * Double(i) / Double(sampleRate)))
        }

        let result = try extractor.process(samples: samples)

        XCTAssertEqual(result.count, 80, "Should have 80 mel bins")
        XCTAssertGreaterThan(result[0].count, 0, "Should have time frames")
    }

    func testMLMultiArrayOutput() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)

        let samples = [Float](repeating: 0.1, count: 8000)
        let mlArray = try extractor.processToMLMultiArray(samples: samples)

        XCTAssertEqual(mlArray.shape.count, 3, "Should be 3D [1, nMels, frames]")
        XCTAssertEqual(mlArray.shape[0].intValue, 1, "Batch size should be 1")
        XCTAssertEqual(mlArray.shape[1].intValue, 80, "Should have 80 mel bins")
    }

    func testFrameCount() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)
        let frameCount = extractor.outputFrameCount(for: 8000)
        XCTAssertGreaterThan(frameCount, 0)
    }

    // MARK: - VAD Model Comparison

    func testVADAgainstReference() throws {
        guard let vadRef = referenceData.models["vad_marblenet"] else {
            XCTFail("VAD reference data not found")
            return
        }

        let extractor = NeMoFeatureExtractor(config: .nemoVAD)
        let swiftMel = try extractor.process(samples: referenceData.audio)
        let nemoMel = vadRef.features

        XCTAssertEqual(swiftMel.count, nemoMel.count, "Mel bins count should match")

        let swiftFrames = swiftMel[0].count
        let nemoFrames = nemoMel[0].count
        let minFrames = min(swiftFrames, nemoFrames)

        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var count = 0

        for mel in 0..<swiftMel.count {
            for frame in 0..<minFrames {
                let diff = abs(swiftMel[mel][frame] - nemoMel[mel][frame])
                maxDiff = max(maxDiff, diff)
                totalDiff += diff
                count += 1
            }
        }

        let avgDiff = totalDiff / Float(count)

        print("VAD comparison:")
        print("  Swift mel shape: [\(swiftMel.count), \(swiftFrames)]")
        print("  NeMo mel shape: [\(nemoMel.count), \(nemoFrames)]")
        print("  Max diff: \(maxDiff)")
        print("  Avg diff: \(avgDiff)")

        // With pre-emphasis and correct window, should be much closer
        XCTAssertLessThan(maxDiff, 1.0, "Max diff should be < 1.0")
        XCTAssertLessThan(avgDiff, 0.1, "Avg diff should be < 0.1")
    }

    // MARK: - Speaker Model Comparison

    func testSpeakerAgainstReference() throws {
        guard let speakerRef = referenceData.models["speaker_titanet"] else {
            XCTFail("Speaker reference data not found")
            return
        }

        let extractor = NeMoFeatureExtractor(config: .nemoSpeaker)
        let swiftMel = try extractor.process(samples: referenceData.audio)
        let nemoMel = speakerRef.features

        XCTAssertEqual(swiftMel.count, nemoMel.count, "Mel bins count should match")

        let swiftFrames = swiftMel[0].count
        let nemoFrames = nemoMel[0].count
        let minFrames = min(swiftFrames, nemoFrames)

        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var count = 0

        for mel in 0..<swiftMel.count {
            for frame in 0..<minFrames {
                let diff = abs(swiftMel[mel][frame] - nemoMel[mel][frame])
                maxDiff = max(maxDiff, diff)
                totalDiff += diff
                count += 1
            }
        }

        let avgDiff = totalDiff / Float(count)

        print("Speaker comparison:")
        print("  Swift mel shape: [\(swiftMel.count), \(swiftFrames)]")
        print("  NeMo mel shape: [\(nemoMel.count), \(nemoFrames)]")
        print("  Max diff: \(maxDiff)")
        print("  Avg diff: \(avgDiff)")

        XCTAssertLessThan(maxDiff, 1.0, "Max diff should be < 1.0")
        XCTAssertLessThan(avgDiff, 0.1, "Avg diff should be < 0.1")
    }

    // MARK: - ASR Model Comparison

    func testASRAgainstReference() throws {
        guard let asrRef = referenceData.models["asr_parakeet"] else {
            XCTFail("ASR reference data not found")
            return
        }

        let extractor = NeMoFeatureExtractor(config: .nemoASR)
        let swiftMel = try extractor.process(samples: referenceData.audio)
        let nemoMel = asrRef.features

        XCTAssertEqual(swiftMel.count, nemoMel.count, "Mel bins count should match")

        let swiftFrames = swiftMel[0].count
        let nemoFrames = nemoMel[0].count
        let minFrames = min(swiftFrames, nemoFrames)

        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var count = 0

        for mel in 0..<swiftMel.count {
            for frame in 0..<minFrames {
                let diff = abs(swiftMel[mel][frame] - nemoMel[mel][frame])
                maxDiff = max(maxDiff, diff)
                totalDiff += diff
                count += 1
            }
        }

        let avgDiff = totalDiff / Float(count)

        print("ASR comparison:")
        print("  Swift mel shape: [\(swiftMel.count), \(swiftFrames)]")
        print("  NeMo mel shape: [\(nemoMel.count), \(nemoFrames)]")
        print("  Max diff: \(maxDiff)")
        print("  Avg diff: \(avgDiff)")

        XCTAssertLessThan(maxDiff, 1.0, "Max diff should be < 1.0")
        XCTAssertLessThan(avgDiff, 0.1, "Avg diff should be < 0.1")
    }

    // MARK: - Config Tests

    func testConfigPresets() throws {
        let vadConfig = MelSpectrogramConfig.nemoVAD
        XCTAssertEqual(vadConfig.sampleRate, 16000)
        XCTAssertEqual(vadConfig.nMels, 80)
        XCTAssertEqual(vadConfig.nFFT, 512)
        XCTAssertEqual(vadConfig.hopLength, 160)
        XCTAssertEqual(vadConfig.normalization, .none)
        XCTAssertEqual(vadConfig.preemph, 0.97)
        XCTAssertEqual(vadConfig.padTo, 2)

        let asrConfig = MelSpectrogramConfig.nemoASR
        XCTAssertEqual(asrConfig.normalization, .perFeature)
        XCTAssertEqual(asrConfig.preemph, 0.97)
        XCTAssertEqual(asrConfig.padTo, 0)

        let speakerConfig = MelSpectrogramConfig.nemoSpeaker
        XCTAssertEqual(speakerConfig.normalization, .perFeature)
        XCTAssertEqual(speakerConfig.preemph, 0.97)
        XCTAssertEqual(speakerConfig.padTo, 16)
    }

    // MARK: - Edge Cases

    func testEmptyInputThrows() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)
        let emptySamples: [Float] = []

        XCTAssertThrowsError(try extractor.process(samples: emptySamples)) { error in
            XCTAssertTrue(error is NeMoFeatureExtractorError)
        }
    }

    func testVeryShortInputThrows() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)
        // 100 samples is less than 1 hop (160), so validFrames = 0
        let shortSamples = [Float](repeating: 0.1, count: 100)

        XCTAssertThrowsError(try extractor.process(samples: shortSamples)) { error in
            XCTAssertTrue(error is NeMoFeatureExtractorError)
        }
    }

    func testMinimumValidInput() throws {
        let extractor = NeMoFeatureExtractor(config: .nemoVAD)
        // Need at least 160 samples for 1 valid frame (hopLength = 160)
        let minSamples = [Float](repeating: 0.1, count: 160)

        let result = try extractor.process(samples: minSamples)
        XCTAssertEqual(result.count, 80, "Should have 80 mel bins")
        XCTAssertGreaterThan(result[0].count, 0, "Should have at least 1 frame")
    }
}
