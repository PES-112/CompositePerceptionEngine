import ARKit
import SwiftUI

@main
struct CPEWalkApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// Built for VoiceOver first: one large start / stop control, the last alert as text, and an
/// optional visual overlay for sighted testers. Hold the phone upright at chest height, camera out.
struct ContentView: View {
    @StateObject private var pipe = WalkPipeline()
    @State private var showOverlay = false

    var body: some View {
        VStack(spacing: 16) {
            Text("CPE Walk").font(.largeTitle.bold()).accessibilityAddTraits(.isHeader)
            Text(pipe.status).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)

            if showOverlay && pipe.running {
                ZStack {
                    ARPreview(session: pipe.session)
                    OverlayView(overlays: pipe.overlays, imageSize: pipe.imageSize)
                }
                .aspectRatio(pipe.imageSize.width / pipe.imageSize.height, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .accessibilityHidden(true)
            } else {
                Spacer()
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Last alert").font(.caption).foregroundStyle(.secondary)
                Text(pipe.lastPhrase.isEmpty ? "—" : pipe.lastPhrase).font(.title2.weight(.semibold))
                Text("Surface: \(pipe.surface)   ·   \(String(format: "%.0f", pipe.latencyMs)) ms per frame")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            Toggle("Surface reminders when quiet", isOn: $pipe.surfaceReminders)
            Toggle("Show camera overlay", isOn: $showOverlay)

            Button(action: toggle) {
                Text(pipe.running ? "Stop" : "Start")
                    .font(.title.bold())
                    .frame(maxWidth: .infinity, minHeight: 96)
            }
            .buttonStyle(.borderedProminent)
            .tint(pipe.running ? .red : .green)
            .accessibilityHint(pipe.running ? "Stops hazard alerts" : "Starts hazard alerts. Hold the phone upright, camera facing forward.")
        }
        .padding()
        .onDisappear { pipe.stop() }
    }

    private func toggle() {
        if pipe.running {
            pipe.stop()
            UIApplication.shared.isIdleTimerDisabled = false
        } else {
            pipe.start()
            UIApplication.shared.isIdleTimerDisabled = true     // the screen must not lock mid-walk
        }
    }
}

/// The camera feed ARKit is already running (no second capture session).
struct ARPreview: UIViewRepresentable {
    let session: ARSession
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView()
        v.session = session
        v.automaticallyUpdatesLighting = false
        return v
    }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct OverlayView: View {
    let overlays: [WalkPipeline.Overlay]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let sx = geo.size.width / imageSize.width, sy = geo.size.height / imageSize.height
            ForEach(Array(overlays.enumerated()), id: \.offset) { _, o in
                let r = CGRect(x: o.box.minX * sx, y: o.box.minY * sy, width: o.box.width * sx, height: o.box.height * sy)
                let c: Color = o.color == 2 ? .red : o.color == 1 ? .orange : .gray
                Rectangle().stroke(c, lineWidth: o.color > 0 ? 3 : 1.5)
                    .frame(width: r.width, height: r.height).position(x: r.midX, y: r.midY)
                Text(o.label).font(.caption2.bold()).foregroundStyle(c)
                    .position(x: r.minX + 40, y: max(8, r.minY - 8))
            }
        }
    }
}
