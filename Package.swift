// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "NeMoFeatureExtractor",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "NeMoFeatureExtractor",
            targets: ["NeMoFeatureExtractor"]
        ),
    ],
    targets: [
        .target(
            name: "NeMoFeatureExtractor",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "NeMoFeatureExtractorTests",
            dependencies: ["NeMoFeatureExtractor"],
            resources: [
                .process("Resources")
            ]
        ),
    ]
)
