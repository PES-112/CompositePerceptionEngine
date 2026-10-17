import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var runner = TimingRunner()
    @State private var sustainedMinutes = 30.0

    var body: some View {
        NavigationStack {
            Form {
                Section("Bundle") {
                    Text(runner.bundleSummary).font(.footnote).textSelection(.enabled)
                }
                Section("Runs") {
                    Button("Benchmark: warm-up + 10 interleaved blocks") { runner.start(.benchmark) }
                        .disabled(runner.running)
                    VStack(alignment: .leading) {
                        Text("Sustained run: \(Int(sustainedMinutes)) min at 15 FPS, ARKit on")
                        Slider(value: $sustainedMinutes, in: 5...40, step: 5)
                    }
                    Button("Start sustained run") { runner.start(.sustained(minutes: Int(sustainedMinutes))) }
                        .disabled(runner.running)
                    Button("Stop", role: .destructive) { runner.stop() }
                        .disabled(!runner.running)
                }
                Section("Status") {
                    Text(runner.status).font(.system(.body, design: .monospaced))
                    if !runner.summary.isEmpty {
                        Text(runner.summary).font(.system(.footnote, design: .monospaced)).textSelection(.enabled)
                    }
                    if let url = runner.lastCSV {
                        ShareLink(item: url) { Label("Share \(url.lastPathComponent)", systemImage: "square.and.arrow.up") }
                    }
                }
                Section {
                    Text("Keep the phone unplugged, screen on, at room temperature. CSVs are also in Files › On My iPhone › CPE Timing.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("CPE Timing")
        }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
    }
}
