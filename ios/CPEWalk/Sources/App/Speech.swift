import AVFoundation

/// Speaks alert phrases with the on-device voice, panned toward the object's bearing (constant-power,
/// full pan at ±60°, as src/narration/spatial.py). An override cuts whatever is playing; any other
/// phrase replaces a phrase still playing only if that one was a surface reminder (the announcer
/// already keeps hazards from overlapping).
final class Speech: NSObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let synth = AVSpeechSynthesizer()
    private let voice = AVSpeechSynthesisVoice(language: "en-US")
    private var format: AVAudioFormat?
    private var generation = 0
    static let panFullDeg = 60.0

    override init() {
        super.init()
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
        engine.attach(player)
    }

    private func ensureRunning(_ f: AVAudioFormat) {
        if format != f {
            engine.stop()
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: f)
            format = f
        }
        if !engine.isRunning { try? engine.start() }
    }

    static func pan(_ bearingDeg: Double) -> Float { Float(max(-1, min(1, bearingDeg / panFullDeg))) }

    func say(_ text: String, bearingDeg: Double, interrupt: Bool) {
        generation += 1
        let gen = generation
        if interrupt || player.isPlaying {
            synth.stopSpeaking(at: .immediate)
            player.stop()
        }
        let u = AVSpeechUtterance(string: text)
        u.voice = voice
        u.rate = 0.56                     // a little faster than default: alerts must be short
        player.pan = Self.pan(bearingDeg)
        synth.write(u) { [weak self] buffer in
            guard let self, gen == self.generation, let pcm = buffer as? AVAudioPCMBuffer, pcm.frameLength > 0 else { return }
            DispatchQueue.main.async {
                guard gen == self.generation else { return }
                self.ensureRunning(pcm.format)
                self.player.scheduleBuffer(pcm)
                if !self.player.isPlaying { self.player.play() }
            }
        }
    }

    func stop() {
        generation += 1
        synth.stopSpeaking(at: .immediate)
        player.stop()
    }
}
