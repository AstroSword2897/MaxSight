//
// MaxSightModels.swift
// Add to your app target. Add the 7 .mlpackage files (amblyopia, amd, color_blindness, cvi, glaucoma, retinitis_pigmentosa, strabismus) to the target.
// Condition, ModelManager, and ModelRunner match your ContentView and openImagesV6Names() flow.
//

import Foundation
import CoreML
import UIKit
import SwiftUI

// MARK: - Condition (7 conditions — one model per case)

enum Condition: String, CaseIterable, Identifiable {
    case amblyopia
    case amd
    case colorBlindness = "color_blindness"
    case cvi
    case glaucoma
    case retinitisPigmentosa = "retinitis_pigmentosa"
    case strabismus

    var id: String { rawValue }

    var displayName: String {
        rawValue.replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
}

// MARK: - ModelManager (load 7 CoreML models by condition)

final class ModelManager {
    static let shared = ModelManager()

    private(set) var loadedModels: [Condition: MLModel] = [:]

    private init() {}

    func loadModel(for condition: Condition) {
        guard let url = Bundle.main.url(forResource: condition.rawValue, withExtension: "mlmodelc") else {
            return
        }
        do {
            let config = MLModelConfiguration()
            let model = try MLModel(contentsOf: url, configuration: config)
            loadedModels[condition] = model
        } catch {
            // Load failed for condition.rawValue
        }
    }
}

// MARK: - ModelRunner (run model; outputs are output_0, output_1, …)

final class ModelRunner {
    static let shared = ModelRunner()

    private init() {}

    func runModel(on image: UIImage, condition: Condition) -> String {
        guard let model = ModelManager.shared.loadedModels[condition] else {
            return "Model not loaded"
        }
        guard let pixelBuffer = image.pixelBuffer() else {
            return "Invalid image"
        }
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let prediction = try model.prediction(from: input)
            return summaryFromPrediction(prediction)
        } catch {
            return "Inference failed"
        }
    }

    private func summaryFromPrediction(_ prediction: MLFeatureProvider) -> String {
        for name in prediction.featureNames.sorted() where name.hasPrefix("output_") {
            guard let value = prediction.featureValue(for: name),
                  let multi = value.multiArrayValue else { continue }
            let count = multi.count
            if count >= 80 {
                var maxIdx = 0
                var maxVal: Float = -Float.greatestFiniteMagnitude
                for i in 0..<min(80, count) {
                    let v = multi[i].floatValue
                    if v > maxVal { maxVal = v; maxIdx = i }
                }
                return "Top class: \(maxIdx)"
            }
            return "Output \(name): \(count) values"
        }
        return "Prediction OK"
    }
}

// MARK: - UIImage → CVPixelBuffer (for CoreML input)

extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        guard let cgImage = cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelBufferFormatType_32ARGB,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

// MARK: - Usage (blends with your existing ContentView)

/*
 Ensure you have openImagesV6Names() in another file, e.g.:

 func openImagesV6Names() -> [String] {
     guard let url = Bundle.main.url(forResource: "manifest", withExtension: "txt", subdirectory: "open_images_v6"),
           let raw = try? String(contentsOf: url, encoding: .utf8) else { return [] }
     return raw.split(separator: "\n").map { String($0).replacingOccurrences(of: ".jpg", with: "") }
 }

 ContentView that uses the 7 conditions and models:

 struct ContentView: View {
     @State private var selectedCondition: Condition = .amblyopia
     @State private var selectedImage: String = "example1"
     @State private var predictionText: String = "Loading..."
     @State private var modelsLoaded = false

     let datasetImages = openImagesV6Names()

     var body: some View {
         GeometryReader { geometry in
             VStack(spacing: 12) {
                 VStack(spacing: 8) {
                     Picker("Condition", selection: $selectedCondition) {
                         ForEach(Condition.allCases) { Text($0.displayName).tag($0) }
                     }
                     .pickerStyle(.segmented)

                     Picker("Dataset Image", selection: $selectedImage) {
                         ForEach(datasetImages, id: \.self) { Text($0) }
                     }
                     .pickerStyle(.menu)

                     Button("Reset Image") {
                         if let random = datasetImages.randomElement() {
                             selectedImage = random
                             runModel()
                         }
                     }
                     .disabled(!modelsLoaded)
                 }
                 .padding(.horizontal)

                 VStack(spacing: 8) {
                     PanelView(title: "Reality View", imageName: selectedImage, overlayText: nil)
                     PanelView(title: "Condition View", imageName: selectedImage, overlayText: nil)
                     PanelView(title: "Awareness View", imageName: selectedImage, overlayText: predictionText)
                 }
                 .frame(height: geometry.size.height * 0.75)
             }
             .frame(width: geometry.size.width, height: geometry.size.height)
         }
         .onAppear { preloadModels() }
         .onChange(of: selectedImage) { _, _ in runModel() }
         .onChange(of: selectedCondition) { _, _ in runModel() }
     }

     func preloadModels() {
         DispatchQueue.global(qos: .userInitiated).async {
             for condition in Condition.allCases {
                 ModelManager.shared.loadModel(for: condition)
             }
             DispatchQueue.main.async {
                 modelsLoaded = true
                 runModel()
             }
         }
     }

     func runModel() {
         guard modelsLoaded else {
             predictionText = "Models loading..."
             return
         }
         guard let uiImage = UIImage(named: selectedImage) else {
             predictionText = "Image not found"
             return
         }
         let resized = uiImage.resized(to: CGSize(width: 224, height: 224))
         predictionText = ModelRunner.shared.runModel(on: resized, condition: selectedCondition)
     }
 }

 struct PanelView: View {
     let title: String
     let imageName: String
     let overlayText: String?

     var body: some View {
         VStack {
             Text(title).font(.headline)
             Image(imageName)
                 .resizable()
                 .scaledToFit()
                 .border(Color.gray)
                 .overlay(
                     overlayText.map {
                         Text($0)
                             .foregroundColor(.white)
                             .padding(6)
                             .background(Color.black.opacity(0.6))
                             .cornerRadius(6)
                             .padding(8)
                     },
                     alignment: .bottomLeading
                 )
         }
         .background(Color(white: 0.95))
         .cornerRadius(8)
         .frame(maxWidth: .infinity)
     }
 }

 extension UIImage {
     func resized(to size: CGSize) -> UIImage {
         UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
         draw(in: CGRect(origin: .zero, size: size))
         let resized = UIGraphicsGetImageFromCurrentImageContext()
         UIGraphicsEndImageContext()
         return resized ?? self
     }
 }
 */
