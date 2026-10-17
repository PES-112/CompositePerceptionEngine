import CoreML
import Foundation
import os
import UIKit

/// What a run does. Benchmark: a warm-up block (discarded in analysis), then 10 paced blocks of
/// 150 frames. Sustained: one paced loop for `minutes`. Frames and depth-derived GRU inputs come
/// from the bundled SANPO data; the phone's own camera and LiDAR are not used (the product's depth
/// sensor is an external LiDAR).
enum RunMode {
    case benchmark
    case sustained(minutes: Int)

    var name: String {
        switch self {
        case .benchmark: return "benchmark"
        case .sustained(let m): return "sustained\(m)min"
        }
    }
}

/// One timed iteration, one CSV row.
struct Sample {
    var block: String
    var iter: Int
    var frame: Int
    var t: Double
    var preMs: Double
    var detMs: Double
    var detPostMs: Double
    var gruMs: Double
    var totalMs: Double
    var thermal: Int
    var battery: Float

    static let header = "block,iter,frame,t_s,pre_ms,det_ms,det_post_ms,gru_ms,total_ms,thermal_state,battery"
    var csv: String {
        String(format: "%@,%d,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f",
               block, iter, frame, t, preMs, detMs, detPostMs, gruMs, totalMs, thermal, battery)
    }
}

@MainActor
final class TimingRunner: ObservableObject {
    @Published var status = "Idle"
    @Published var summary = ""
    @Published var running = false
    @Published var lastCSV: URL?
    let bundleSummary: String

    private var worker: Task<Void, Never>?

    init() {
        bundleSummary = (try? Workload.load().description) ?? "Bundle incomplete: run prepare_bundle.py before building."
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    func start(_ mode: RunMode) {
        guard !running else { return }
        running = true
        summary = ""
        worker = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let workload = try Workload.load()
                let url = try await Self.run(mode, workload: workload) { text in
                    await MainActor.run { self?.status = text }
                }
                let report = try Summary.of(csv: url)
                await MainActor.run {
                    self?.lastCSV = url
                    self?.summary = report
                    self?.status = "Done: \(url.lastPathComponent)"
                }
            } catch {
                await MainActor.run { self?.status = "Error: \(error.localizedDescription)" }
            }
            await MainActor.run { self?.running = false }
        }
    }

    func stop() { worker?.cancel() }

    // MARK: - The timed loop (off the main actor)

    nonisolated static func run(_ mode: RunMode, workload: Workload,
                                progress: @escaping (String) async -> Void) async throws -> URL {
        let signposter = OSSignposter(subsystem: "edu.pes.cpe.timing", category: .pointsOfInterest)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = docs.appendingPathComponent("cpe_timing_\(mode.name)_\(stamp).csv")
        var out = "# device=\(Device.model) os=\(Device.os) detector=\(workload.detectorName) gru=\(workload.gruName) " +
                  "frames=\(workload.frames.count) mode=\(mode.name)\n" + Sample.header + "\n"

        // (block label, iterations, paced at 15 FPS)
        var plan: [(String, Int, Bool)] = []
        switch mode {
        case .benchmark:
            plan.append(("warmup", 60, false))
            for b in 0..<10 { plan.append(("block\(b)", 150, true)) }
        case .sustained(let minutes):
            plan.append(("warmup", 60, false))
            plan.append(("sustained", minutes * 60 * 15, true))
        }

        let start = DispatchTime.now().uptimeNanoseconds
        var iter = 0
        var battery: Float = -1
        for (label, count, paced) in plan {
            var next = DispatchTime.now().uptimeNanoseconds
            for i in 0..<count {
                if Task.isCancelled { break }
                let frame = iter % workload.frames.count
                let id = signposter.makeSignpostID()
                let state = signposter.beginInterval("frame", id: id)
                let s = try workload.step(frame: frame, signposter: signposter)
                signposter.endInterval("frame", state)
                if i % 15 == 0 {
                    battery = await MainActor.run { UIDevice.current.batteryLevel }
                }
                let t = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                out += Sample(block: label, iter: iter, frame: frame, t: t, preMs: s.pre, detMs: s.det,
                              detPostMs: s.post, gruMs: s.gru, totalMs: s.total,
                              thermal: ProcessInfo.processInfo.thermalState.rawValue, battery: battery).csv + "\n"
                iter += 1
                if i % 75 == 0 {
                    await progress(String(format: "%@ %d/%d  total %.1f ms  thermal %d", label, i, count, s.total,
                                          ProcessInfo.processInfo.thermalState.rawValue))
                }
                if i % 900 == 0 { try out.write(to: url, atomically: true, encoding: .utf8) }   // survive a kill
                if paced {
                    next += 66_666_667   // 15 FPS
                    let now = DispatchTime.now().uptimeNanoseconds
                    if next > now { try await Task.sleep(nanoseconds: next - now) } else { next = now }
                }
            }
            if Task.isCancelled { break }
        }
        try out.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}

// MARK: - Workload: detector + GRU over bundled SANPO frames and recorded GRU inputs

struct StageMs { var pre: Double; var det: Double; var post: Double; var gru: Double; var total: Double }

enum BundleError: LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self { case .missing(let what): return "Missing from the app bundle: \(what)" }
    }
}

final class Workload: @unchecked Sendable {
    let frames: [URL]
    let detector: MLModel
    let gru: MLModel
    let detectorName: String
    let gruName: String
    private let imageInput: String
    private let imageSize: (Int, Int)
    private let extraInputs: [String: MLFeatureValue]
    private let gruX: MLMultiArray      // [1, N, T, F], refilled per frame
    private let gruCls: MLMultiArray    // [1, N]
    private let recordedX: [Float]      // [M, N, T, F] from the SANPO test split (prepare_bundle.py)
    private let recordedCls: [Int32]    // [M, N]
    private let perFrameX: Int
    private let perFrameCls: Int
    private let recordedFrames: Int

    var description: String {
        "\(frames.count) frames, detector \(detectorName) (\(imageSize.0)x\(imageSize.1)), GRU \(gruName), " +
        "\(recordedFrames) recorded GRU inputs"
    }

    static func load() throws -> Workload { try Workload() }

    private init() throws {
        let res = Bundle.main.resourceURL!.appendingPathComponent("Resources")
        let frameDir = res.appendingPathComponent("Frames")
        frames = ((try? FileManager.default.contentsOfDirectory(at: frameDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !frames.isEmpty else { throw BundleError.missing("Resources/Frames/*.jpg") }

        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let detURL = Bundle.main.url(forResource: "CPEDetector", withExtension: "mlmodelc") else {
            throw BundleError.missing("CPEDetector.mlpackage") }
        guard let gruURL = Bundle.main.url(forResource: "CPEArrivalGRU", withExtension: "mlmodelc") else {
            throw BundleError.missing("CPEArrivalGRU.mlpackage") }
        detector = try MLModel(contentsOf: detURL, configuration: config)
        gru = try MLModel(contentsOf: gruURL, configuration: config)
        detectorName = detector.modelDescription.metadata[.description] as? String ?? "CPEDetector"
        gruName = gru.modelDescription.metadata[.description] as? String ?? "CPEArrivalGRU"

        // The detector's image input and any scalar inputs an NMS pipeline adds (thresholds).
        var image: (String, Int, Int)?
        var extras: [String: MLFeatureValue] = [:]
        for (name, d) in detector.modelDescription.inputDescriptionsByName {
            if d.type == .image, let c = d.imageConstraint {
                image = (name, c.pixelsWide, c.pixelsHigh)
            } else if d.type == .double && !d.isOptional {
                extras[name] = MLFeatureValue(double: name.lowercased().contains("iou") ? 0.45 : 0.25)
            }
        }
        guard let image else { throw BundleError.missing("an image input on CPEDetector") }
        imageInput = image.0
        imageSize = (image.1, image.2)
        extraInputs = extras

        let xURL = res.appendingPathComponent("gru_x.bin"), cURL = res.appendingPathComponent("gru_cls.bin")
        guard let xData = try? Data(contentsOf: xURL), let cData = try? Data(contentsOf: cURL) else {
            throw BundleError.missing("Resources/gru_x.bin, gru_cls.bin") }
        guard let xShape = gru.modelDescription.inputDescriptionsByName["x"]?.multiArrayConstraint?.shape,
              let cShape = gru.modelDescription.inputDescriptionsByName["cls"]?.multiArrayConstraint?.shape else {
            throw BundleError.missing("GRU inputs x and cls") }
        gruX = try MLMultiArray(shape: xShape, dataType: .float32)
        gruCls = try MLMultiArray(shape: cShape, dataType: .int32)
        perFrameX = xShape.map(\.intValue).reduce(1, *)
        perFrameCls = cShape.map(\.intValue).reduce(1, *)
        recordedX = xData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        recordedCls = cData.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        recordedFrames = min(recordedX.count / perFrameX, recordedCls.count / perFrameCls)
        guard recordedFrames > 0 else { throw BundleError.missing("GRU inputs matching the model's shape") }
    }

    /// One frame: decode + resize (pre), detector (det), touch every output value (post), GRU (gru).
    func step(frame: Int, signposter: OSSignposter) throws -> StageMs {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let pre = signposter.beginInterval("pre")
        guard let buffer = PixelBuffer.from(url: frames[frame], width: imageSize.0, height: imageSize.1) else {
            throw BundleError.missing("decodable frame \(frames[frame].lastPathComponent)") }
        signposter.endInterval("pre", pre)
        let t1 = DispatchTime.now().uptimeNanoseconds

        let det = signposter.beginInterval("detector")
        var inputs = extraInputs
        inputs[imageInput] = MLFeatureValue(pixelBuffer: buffer)
        let out = try detector.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        signposter.endInterval("detector", det)
        let t2 = DispatchTime.now().uptimeNanoseconds

        var checksum: Float = 0                        // read the outputs so they can't be skipped lazily
        for name in out.featureNames {
            if let a = out.featureValue(for: name)?.multiArrayValue {
                let n = a.count
                for i in stride(from: 0, to: n, by: max(1, n / 4096)) { checksum += a[i].floatValue }
            }
        }
        let t3 = DispatchTime.now().uptimeNanoseconds

        let g = signposter.beginInterval("gru")
        let k = frame % recordedFrames
        recordedX.withUnsafeBufferPointer { src in
            gruX.dataPointer.assumingMemoryBound(to: Float.self)
                .update(from: src.baseAddress! + k * perFrameX, count: perFrameX)
        }
        recordedCls.withUnsafeBufferPointer { src in
            gruCls.dataPointer.assumingMemoryBound(to: Int32.self)
                .update(from: src.baseAddress! + k * perFrameCls, count: perFrameCls)
        }
        let p = try gru.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": gruX, "cls": gruCls]))
        if let probs = p.featureValue(for: "p")?.multiArrayValue { checksum += probs[0].floatValue }
        signposter.endInterval("gru", g)
        let t4 = DispatchTime.now().uptimeNanoseconds
        if checksum.isNaN { NSLog("cpe: NaN output at frame %d", frame) }

        let ms = { (a: UInt64, b: UInt64) in Double(b - a) / 1e6 }
        return StageMs(pre: ms(t0, t1), det: ms(t1, t2), post: ms(t2, t3), gru: ms(t3, t4), total: ms(t0, t4))
    }
}

enum PixelBuffer {
    /// Decode a JPEG and draw it, aspect-filled, into a BGRA buffer of the detector's input size.
    static func from(url: URL, width: Int, height: Int) -> CVPixelBuffer? {
        guard let image = UIImage(contentsOfFile: url.path)?.cgImage else { return nil }
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &buffer) == kCVReturnSuccess, let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        let scale = max(Double(width) / Double(image.width), Double(height) / Double(image.height))
        let w = Double(image.width) * scale, h = Double(image.height) * scale
        ctx.draw(image, in: CGRect(x: (Double(width) - w) / 2, y: (Double(height) - h) / 2, width: w, height: h))
        return buffer
    }
}

// MARK: - Summary shown in the app (the full analysis is ios/CPETiming/analyze_timing.py)

enum Summary {
    static func of(csv url: URL) throws -> String {
        let rows = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
            .filter { !$0.hasPrefix("#") && !$0.hasPrefix("block") }.map { $0.split(separator: ",") }
        var groups: [String: [Double]] = [:]
        for r in rows where r.count > 8 && r[0] != "warmup" {
            groups[r[0] == "sustained" ? "sustained" : "benchmark", default: []].append(Double(r[8]) ?? .nan)
        }
        return groups.keys.sorted().map { key in
            let v = groups[key]!.sorted()
            let q = { (p: Double) in v[min(v.count - 1, Int(p * Double(v.count - 1)))] }
            return String(format: "%@ total ms: median %.1f  p95 %.1f  p99 %.1f  (n=%d)", key, q(0.5), q(0.95), q(0.99), v.count)
        }.joined(separator: "\n")
    }
}

enum Device {
    static var model: String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
    }
    static var os: String { ProcessInfo.processInfo.operatingSystemVersionString.replacingOccurrences(of: ",", with: ";") }
}
