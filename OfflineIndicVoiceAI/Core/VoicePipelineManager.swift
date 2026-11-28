//
//  VoicePipelineManager.swift
//  OfflineIndicVoiceAI
//
//  100% Offline, Local Voice-to-Voice Architecture for iOS:
//  AVAudioEngine (16kHz, 16-bit, Mono PCM)
//    -> Sherpa-ONNX (Silero VAD + IndicConformer INT8 ONNX ASR)
//    -> LiteRT-LM (Gemma 4 E2B-IT .litertlm via Metal GPU, Transliterated Cross-Lingual Bridge)
//    -> Transliteration Stream Chunker & Native-Script Sanitizer
//    -> ONNX Runtime (Kokoro 82M INT8/FP16 ONNX TTS, 24kHz Float32)
//    -> AVAudioPlayerNode (Gapless Local Audio Playback)
//
//  Memory Safety: Designed explicitly around iOS Jetsam ceilings using
//  `os_proc_available_memory()`, `autoreleasepool` tensor scoping, memory-mapped
//  voice embeddings, bounded ring/stream buffers, KV-cache turn caps, and
//  tiered engine eviction under OS memory pressure.
//

import Accelerate
import AVFoundation
import Combine
import Darwin
import Foundation
import os
import UIKit

#if canImport(Speech)
import Speech
#endif

#if canImport(SherpaOnnx)
import SherpaOnnx
#elseif canImport(sherpa_onnx)
import sherpa_onnx
#endif

#if canImport(LiteRTLM)
import LiteRTLM
#endif

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#elseif canImport(onnxruntime)
import onnxruntime
#endif

// MARK: - Supported Languages & Interaction Modes

public enum PipelineLanguage: String, CaseIterable, Identifiable, Sendable {
    case auto = "auto"
    case punjabi = "pa"
    case hindi = "hi"
    case english = "en"
    case bengali = "bn"
    case tamil = "ta"
    case telugu = "te"
    case marathi = "mr"
    case gujarati = "gu"
    case kannada = "kn"

    public var id: String { rawValue }

    public static var outputCases: [PipelineLanguage] {
        allCases.filter { $0 != .auto }
    }

    public var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .punjabi: return "Punjabi"
        case .hindi: return "Hindi"
        case .english: return "English"
        case .bengali: return "Bengali"
        case .tamil: return "Tamil"
        case .telugu: return "Telugu"
        case .marathi: return "Marathi"
        case .gujarati: return "Gujarati"
        case .kannada: return "Kannada"
        }
    }

    public var nativeScriptLabel: String {
        switch self {
        case .auto: return "Auto"
        case .punjabi: return "ਪੰਜਾਬੀ"
        case .hindi: return "हिन्दी"
        case .english: return "English"
        case .bengali: return "বাংলা"
        case .tamil: return "தமிழ்"
        case .telugu: return "తెలుగు"
        case .marathi: return "मराठी"
        case .gujarati: return "ગુજરાતી"
        case .kannada: return "ಕನ್ನಡ"
        }
    }

    public var shortChipLabel: String {
        switch self {
        case .auto: return "Auto"
        case .english: return "English"
        default: return "\(displayName) (\(nativeScriptLabel))"
        }
    }
}

public enum AssistantInteractionMode: String, CaseIterable, Identifiable, Sendable {
    case translate = "translate"
    case smartAnswer = "smartAnswer"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .translate: return "Translate & Learn"
        case .smartAnswer: return "Smart AI Answer"
        }
    }

    public var shortTitle: String {
        switch self {
        case .translate: return "Translate"
        case .smartAnswer: return "AI Answer"
        }
    }

    public var iconName: String {
        switch self {
        case .translate: return "character.book.closed"
        case .smartAnswer: return "brain.head.profile"
        }
    }
}

// MARK: - Permanent System Instructions (Gemma 4 E2B-IT)

public enum GemmaSystemPrompt {
    public static let transliterationBridge: String = build(
        input: .auto,
        output: .punjabi,
        mode: .smartAnswer
    )

    public static func build(
        input: PipelineLanguage,
        output: PipelineLanguage,
        mode: AssistantInteractionMode
    ) -> String {
        let resolvedOutput = (output == .auto) ? .english : output
        let inputDesc = (input == .auto)
            ? "the user's spoken language (which may be Punjabi, Hindi, English, or another Indian language)"
            : "\(input.displayName) (\(input.nativeScriptLabel))"

        let purityRule: String
        switch resolvedOutput {
        case .english:
            purityRule = """
            Respond ONLY in pure, natural, grammatically correct English. Never mix Hindi, Punjabi, or any non-Latin script.
            Output ONLY the plain English text with zero labels, zero markdown, and zero filler.
            """
        case .hindi:
            purityRule = """
            Respond ONLY in pure Hindi. STRICTLY FORBIDDEN: Never use Punjabi words such as "tusi", "mainu", "vadiya", "aan", "sakdaan", "ki gall", or "kiven".
            Output EXACTLY two lines:
            NATIVE: <the Hindi response in pure Devanagari script (हिन्दी)>
            SPEAK: <the EXACT same Hindi response written phonetically in simple English Latin letters for Text-to-Speech, e.g., "Haan ji, main aapko sun sakta hoon. Bataiye main aapki kya madad karoon?">
            """
        case .punjabi:
            purityRule = """
            Respond ONLY in pure natural Punjabi. STRICTLY FORBIDDEN: Never mix Hindi words such as "aap", "kya", "hoon", "hain", "sakta", "bataiye", or "aapki".
            Output EXACTLY two lines:
            NATIVE: <the Punjabi response in pure Gurmukhi script (ਪੰਜਾਬੀ)>
            SPEAK: <the EXACT same Punjabi response written phonetically in simple English Latin letters for Text-to-Speech, e.g., "Haan ji, main tuhadi gal sun sakda haan. Daso main ki madad karaan?">
            """
        default:
            purityRule = """
            Respond ONLY in pure natural \(resolvedOutput.displayName) (\(resolvedOutput.nativeScriptLabel)). Do NOT mix words from Hindi, Punjabi, or other languages.
            Output EXACTLY two lines:
            NATIVE: <the \(resolvedOutput.displayName) response in pure \(resolvedOutput.nativeScriptLabel) script>
            SPEAK: <the EXACT same \(resolvedOutput.displayName) response written phonetically in simple English Latin letters for Text-to-Speech>
            """
        }

        let taskObjective: String
        switch mode {
        case .translate:
            taskObjective = """
            Your ONLY task is ACCURATE VOICE TRANSLATION for language learning.
            The user spoke in \(inputDesc). Note: if the input text looks like phonetic Indic spelling of English words (e.g., "ਆਰ ਯੂ ਏਬਲ ਟੂ ਲੇਸਨ"), understand its spoken meaning accurately.
            Translate what the user said directly and accurately into \(resolvedOutput.displayName). Do NOT answer their question — ONLY translate their sentence into \(resolvedOutput.displayName).
            """
        case .smartAnswer:
            taskObjective = """
            Your task is to be a helpful, concise voice AI tutor and assistant.
            The user spoke in \(inputDesc). Note: if the input text looks like phonetic Indic spelling of English words, understand its spoken meaning accurately.
            Understand the user's question or statement and give an intelligent, direct, conversational answer (1-2 short sentences) in \(resolvedOutput.displayName).
            """
        }

        return """
        <system_instructions>
            <role>You are an offline voice-to-voice language learning and AI assistant on iPhone.</role>
            <task>\(taskObjective)</task>
            <output_format_rules>
                \(purityRule)
                Keep responses concise (1-2 short sentences) for fast voice playback.
                Do not include conversational filler like "Sure, here is the translation:" and do not use markdown or bullet points.
            </output_format_rules>
        </system_instructions>
        """
    }
}

// MARK: - Configuration & Domain Models

public struct VoicePipelineConfiguration: Sendable {
    public struct ASRConfig: Sendable {
        public let indicConformerModelPath: String
        public let tokensPath: String
        public let sileroVadModelPath: String
        public let sampleRate: Int32
        public let numThreads: Int32
        public let provider: String
        public let vadThreshold: Float
        public let minSilenceDurationSeconds: Float
        public let minSpeechDurationSeconds: Float
        public let maxUtteranceSeconds: Float
        public let scriptDisplayMode: ScriptDisplayMode
        public let inputLanguage: PipelineLanguage

        public init(
            indicConformerModelPath: String,
            tokensPath: String,
            sileroVadModelPath: String,
            sampleRate: Int32 = 16_000,
            numThreads: Int32 = 2,
            provider: String = "cpu",
            vadThreshold: Float = 0.50,
            minSilenceDurationSeconds: Float = 0.45,
            minSpeechDurationSeconds: Float = 0.25,
            maxUtteranceSeconds: Float = 15.0,
            scriptDisplayMode: ScriptDisplayMode = .auto,
            inputLanguage: PipelineLanguage = .auto
        ) {
            self.indicConformerModelPath = indicConformerModelPath
            self.tokensPath = tokensPath
            self.sileroVadModelPath = sileroVadModelPath
            self.sampleRate = sampleRate
            self.numThreads = numThreads
            self.provider = provider
            self.vadThreshold = vadThreshold
            self.minSilenceDurationSeconds = minSilenceDurationSeconds
            self.minSpeechDurationSeconds = minSpeechDurationSeconds
            self.maxUtteranceSeconds = maxUtteranceSeconds
            self.scriptDisplayMode = scriptDisplayMode
            self.inputLanguage = inputLanguage
        }
    }

    public enum ScriptDisplayMode: String, Sendable {
        case auto
        case punjabiGurmukhi
        case nativeASR
    }

    public struct LLMConfig: Sendable {
        public let modelPath: String
        public let cacheDirectory: String
        public let maxContextTokens: Int
        public let topK: Int
        public let topP: Double
        public let temperature: Double
        public let maxTurnsBeforeKVReset: Int
        public let systemPrompt: String

        public init(
            modelPath: String,
            cacheDirectory: String = NSTemporaryDirectory(),
            maxContextTokens: Int = 2048,
            topK: Int = 32,
            topP: Double = 0.90,
            temperature: Double = 0.35,
            maxTurnsBeforeKVReset: Int = 6,
            systemPrompt: String = GemmaSystemPrompt.transliterationBridge
        ) {
            self.modelPath = modelPath
            self.cacheDirectory = cacheDirectory
            self.maxContextTokens = maxContextTokens
            self.topK = topK
            self.topP = topP
            self.temperature = temperature
            self.maxTurnsBeforeKVReset = maxTurnsBeforeKVReset
            self.systemPrompt = systemPrompt
        }
    }

    public struct TTSConfig: Sendable {
        public let kokoroModelPath: String
        public let voicePackPath: String
        public let vocabJSONPath: String
        public let defaultVoiceID: String
        public let speed: Float
        public let outputSampleRate: Double
        public let maxTokenLength: Int
        public let intraOpThreads: Int32
        public let useCoreMLExecutionProvider: Bool

        public init(
            kokoroModelPath: String,
            voicePackPath: String,
            vocabJSONPath: String,
            defaultVoiceID: String = "af_bella",
            speed: Float = 1.0,
            outputSampleRate: Double = 24_000,
            maxTokenLength: Int = 510,
            intraOpThreads: Int32 = 2,
            useCoreMLExecutionProvider: Bool = false
        ) {
            self.kokoroModelPath = kokoroModelPath
            self.voicePackPath = voicePackPath
            self.vocabJSONPath = vocabJSONPath
            self.defaultVoiceID = defaultVoiceID
            self.speed = speed
            self.outputSampleRate = outputSampleRate
            self.maxTokenLength = min(maxTokenLength, 510)
            self.intraOpThreads = intraOpThreads
            self.useCoreMLExecutionProvider = useCoreMLExecutionProvider
        }
    }

    public enum ResidencyPolicy: Sendable {
        /// Keep ASR, Gemma 4, and Kokoro resident concurrently (recommended for >= 8GB RAM devices).
        case alwaysResident
        /// Keep Gemma 4 warm; dynamically page ASR / TTS based on `os_proc_available_memory()`.
        case adaptiveLowMemory
        /// Unload all heavy sessions immediately in background or under `.warning` memory pressure.
        case strictForegroundOnly
    }

    public enum AudioRoutingMode: Sendable {
        /// Uses `.voiceChat` hardware AEC so user can barge in while Kokoro speaks.
        case fullDuplexVoiceChatAEC
        /// Uses `.spokenAudio` + software capture gate during playback for maximum TTS fidelity.
        case halfDuplexGated
    }

    public let asr: ASRConfig
    public let llm: LLMConfig
    public let tts: TTSConfig
    public let residencyPolicy: ResidencyPolicy
    public let audioRoutingMode: AudioRoutingMode
    public let minimumMemoryHeadroomMB: Int
    public let inputLanguage: PipelineLanguage
    public let outputLanguage: PipelineLanguage
    public let interactionMode: AssistantInteractionMode

    public init(
        asr: ASRConfig,
        llm: LLMConfig,
        tts: TTSConfig,
        residencyPolicy: ResidencyPolicy = .adaptiveLowMemory,
        audioRoutingMode: AudioRoutingMode = .halfDuplexGated,
        minimumMemoryHeadroomMB: Int = 450,
        inputLanguage: PipelineLanguage = .auto,
        outputLanguage: PipelineLanguage = .punjabi,
        interactionMode: AssistantInteractionMode = .smartAnswer
    ) {
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.residencyPolicy = residencyPolicy
        self.audioRoutingMode = audioRoutingMode
        self.minimumMemoryHeadroomMB = minimumMemoryHeadroomMB
        self.inputLanguage = inputLanguage
        self.outputLanguage = outputLanguage
        self.interactionMode = interactionMode
    }
}

public enum VoicePipelineState: Equatable, Sendable {
    case uninitialized
    case warmingModels
    case listening
    case transcribing
    case thinking
    case speaking
    case pausedForMemoryPressure
    case error(String)
}

public struct TurnLatencyMetrics: Sendable, Equatable {
    public let asrDecodeMs: Double
    public let llmFirstTokenMs: Double
    public let ttsFirstClauseMs: Double
    public let totalTurnMs: Double
}

public enum VoicePipelineEvent: Sendable {
    case stateChanged(VoicePipelineState)
    case speechActivityChanged(isSpeaking: Bool)
    case transcriptRecognized(String)
    case llmTokenReceived(String)
    case ttsClauseSynthesized(clause: String, sampleCount: Int)
    case turnCompleted(TurnLatencyMetrics)
    case memoryPressureTriggered(availableMB: Int)
}

public enum VoicePipelineError: LocalizedError, Sendable {
    case microphonePermissionDenied
    case audioHardwareUnavailable
    case audioConverterInitFailed
    case modelFileMissing(String)
    case engineNotInitialized(String)
    case invalidTensorShape(String)
    case phonemizationFailed(String)
    case insufficientMemoryHeadroom(availableMB: Int, requiredMB: Int)
    case underlying(String)

    public var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone permission was denied by the user."
        case .audioHardwareUnavailable:
            return "Audio input hardware format is unavailable or has zero channels."
        case .audioConverterInitFailed:
            return "Failed to initialize AVAudioConverter to 16kHz Int16 Mono PCM."
        case .modelFileMissing(let path):
            return "Required offline model asset missing at path: \(path)"
        case .engineNotInitialized(let component):
            return "\(component) has not been initialized."
        case .invalidTensorShape(let details):
            return "Tensor shape mismatch: \(details)"
        case .phonemizationFailed(let text):
            return "Could not phonemize text into Kokoro vocabulary tokens: \(text)"
        case .insufficientMemoryHeadroom(let availableMB, let requiredMB):
            return "Insufficient RAM headroom (\(availableMB) MB available, \(requiredMB) MB required)."
        case .underlying(let message):
            return message
        }
    }
}

// MARK: - Deliverable 5: Jetsam-Aware Memory Budget & Lifecycle Monitor

public enum MemoryPressureLevel: Int, Comparable, Sendable {
    case nominal = 0
    case elevated = 1
    case critical = 2

    public static func < (lhs: MemoryPressureLevel, rhs: MemoryPressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public final class MemoryBudgetManager: @unchecked Sendable {
    public typealias PressureHandler = @Sendable (MemoryPressureLevel, Int) -> Void

    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "MemoryBudget")
    private let monitorQueue = DispatchQueue(label: "com.offlineindic.voice.memory", qos: .utility)
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var notificationTokens: [NSObjectProtocol] = []
    private let onPressure: PressureHandler

    public init(onPressure: @escaping PressureHandler) {
        self.onPressure = onPressure
        startMonitoring()
    }

    deinit {
        stopMonitoring()
    }

    /// Queries the iOS kernel (`os_proc_available_memory`) for exact remaining bytes
    /// before Jetsam terminates the process.
    public static func availableMemoryMB() -> Int {
        let bytes = os_proc_available_memory()
        return Int(bytes / (1024 * 1024))
    }

    public static var totalPhysicalRAMGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    }

    /// Returns true if the device has enough memory headroom to keep all 3 engines resident.
    public func hasHeadroom(minimumRequiredMB: Int) -> Bool {
        Self.availableMemoryMB() >= minimumRequiredMB
    }

    private func startMonitoring() {
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: monitorQueue
        )
        source.setEventHandler { [weak self, weak source] in
            guard let self, let source else { return }
            let data = source.data
            let available = Self.availableMemoryMB()
            let level: MemoryPressureLevel = data.contains(.critical) ? .critical : .elevated
            self.logger.warning("DispatchSource memory pressure \(String(describing: level)), available=\(available) MB")
            self.onPressure(level, available)
        }
        source.resume()
        self.memoryPressureSource = source

        let center = NotificationCenter.default
        let memWarnToken = center.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            let available = Self.availableMemoryMB()
            self?.logger.error("UIApplication didReceiveMemoryWarningNotification, available=\(available) MB")
            self?.onPressure(.critical, available)
        }

        let bgToken = center.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            let available = Self.availableMemoryMB()
            self?.logger.info("App entered background, available=\(available) MB")
            self?.onPressure(.elevated, available)
        }

        notificationTokens = [memWarnToken, bgToken]
    }

    private func stopMonitoring() {
        memoryPressureSource?.cancel()
        memoryPressureSource = nil
        notificationTokens.forEach { NotificationCenter.default.removeObserver($0) }
        notificationTokens.removeAll()
    }
}

// MARK: - Deliverable 2: AVFoundation 16kHz 16-bit Mono PCM Capture & 24kHz Playback

public struct PCM16kChunk: Sendable {
    /// Raw 16,000 Hz, 16-bit signed integer, mono PCM payload.
    public let pcmBytes: Data
    public let frameCount: Int
    public let hostTime: UInt64

    /// Decodes Int16 little-endian PCM into normalized Float32 [-1.0, 1.0] using Accelerate.
    /// Writes into caller-provided `destination` to avoid repeated heap allocations.
    public func copyNormalizedFloat32(into destination: inout [Float]) {
        if destination.count != frameCount {
            destination = [Float](repeating: 0, count: frameCount)
        }
        guard frameCount > 0 else { return }
        pcmBytes.withUnsafeBytes { rawPtr in
            guard let int16Ptr = rawPtr.bindMemory(to: Int16.self).baseAddress else { return }
            vDSP_vflt16(int16Ptr, 1, &destination, 1, vDSP_Length(frameCount))
            var scale: Float = 32768.0
            vDSP_vsdiv(destination, 1, &scale, &destination, 1, vDSP_Length(frameCount))
        }
    }
}

/// Thread-safe, fixed-capacity circular PCM buffer avoiding unbounded growth on audio callbacks.
private final class FixedRingPCMBuffer: @unchecked Sendable {
    private var storage: [Int16]
    private var writeIndex: Int = 0
    private var readIndex: Int = 0
    private var count: Int = 0
    private var lock = os_unfair_lock_s()
    private let capacity: Int

    init(capacitySamples: Int) {
        self.capacity = max(16_000, capacitySamples)
        self.storage = [Int16](repeating: 0, count: self.capacity)
    }

    func append(_ samples: UnsafeBufferPointer<Int16>) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        for sample in samples {
            storage[writeIndex] = sample
            writeIndex = (writeIndex + 1) % capacity
            if count == capacity {
                readIndex = (readIndex + 1) % capacity
            } else {
                count += 1
            }
        }
    }

    func drain(maxSamples: Int) -> Data {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let n = min(count, maxSamples)
        guard n > 0 else { return Data() }

        var data = Data(count: n * MemoryLayout<Int16>.size)
        data.withUnsafeMutableBytes { outRaw in
            guard let outPtr = outRaw.bindMemory(to: Int16.self).baseAddress else { return }
            let firstChunk = min(n, capacity - readIndex)
            storage.withUnsafeBufferPointer { srcPtr in
                guard let base = srcPtr.baseAddress else { return }
                outPtr.update(from: base.advanced(by: readIndex), count: firstChunk)
                if n > firstChunk {
                    outPtr.advanced(by: firstChunk).update(from: base, count: n - firstChunk)
                }
            }
        }
        readIndex = (readIndex + n) % capacity
        count -= n
        return data
    }

    func clear() {
        os_unfair_lock_lock(&lock)
        writeIndex = 0
        readIndex = 0
        count = 0
        os_unfair_lock_unlock(&lock)
    }
}

/// Thread-safe mutable value wrapper for Swift 6 strict concurrency.
private final class LockedBox<Value>: @unchecked Sendable {
    private var value: Value
    private var lock = os_unfair_lock_s()

    init(_ initialValue: Value) {
        self.value = initialValue
    }

    func withLock<R>(_ body: (inout Value) -> R) -> R {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return body(&value)
    }
}

public final class AudioPipelineIO: @unchecked Sendable {
    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "AudioPipelineIO")

    private var audioEngine = AVAudioEngine()
    private var playerNode = AVAudioPlayerNode()
    private let graphQueue = DispatchQueue(label: "com.offlineindic.voice.audiograph")

    private var converter: AVAudioConverter?
    private var reusableConvertedBuffer: AVAudioPCMBuffer?

    private let routingMode: VoicePipelineConfiguration.AudioRoutingMode
    private let targetSampleRate: Double = 16_000
    private let kokoroOutputSampleRate: Double = 24_000

    /// Mandatory capture contract: 16,000 Hz, 16-bit signed integer, Mono PCM.
    public let captureFormat16kInt16Mono: AVAudioFormat
    public let playbackFormat24kFloatMono: AVAudioFormat

    private let ringBuffer = FixedRingPCMBuffer(capacitySamples: 16_000 * 4) // 4-second max ring
    private let continuationBox = LockedBox<AsyncStream<PCM16kChunk>.Continuation?>(nil)
    private let playbackStateLock = NSLock()
    private var pendingPlaybackBuffers: Int = 0
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []
    private var capacityWaiters: [CheckedContinuation<Void, Never>] = []

    private var isCaptureGatedForPlayback = LockedBox<Bool>(false)
    private var isTapInstalled = false
    private var routeObservers: [NSObjectProtocol] = []

    public init(routingMode: VoicePipelineConfiguration.AudioRoutingMode) throws {
        self.routingMode = routingMode
        guard
            let captureFmt = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: targetSampleRate,
                channels: 1,
                interleaved: true
            ),
            let playbackFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: kokoroOutputSampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw VoicePipelineError.audioConverterInitFailed
        }
        self.captureFormat16kInt16Mono = captureFmt
        self.playbackFormat24kFloatMono = playbackFmt
        registerAudioNotifications()
    }

    deinit {
        routeObservers.forEach { NotificationCenter.default.removeObserver($0) }
        stop()
    }

    /// Creates a fresh bounded stream of 16kHz 16-bit Mono PCM chunks for each pipeline session.
    public func makePCM16kStream() -> AsyncStream<PCM16kChunk> {
        AsyncStream(bufferingPolicy: .bufferingNewest(64)) { continuation in
            let previous = continuationBox.withLock { current -> AsyncStream<PCM16kChunk>.Continuation? in
                let old = current
                current = continuation
                return old
            }
            previous?.finish()
        }
    }

    public func requestMicrophonePermission() async -> Bool {
        if #available(iOS 17.0, *) {
            return await AVAudioApplication.requestRecordPermission()
        } else {
            return await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        }
    }

    private func resetEngineAndDeactivateSessionLocked() {
        if isTapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }
        playerNode.stop()
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        converter = nil

        let session = AVAudioSession.sharedInstance()
        try? session.setPreferredSampleRate(0)
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
    }

    public func configureForPlaybackOnly() throws {
        try graphQueue.sync {
            resetEngineAndDeactivateSessionLocked()

            let session = AVAudioSession.sharedInstance()
            do {
                try session.setCategory(.playback, mode: .spokenAudio, options: [])
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            } catch {
                // Fallback to playAndRecord defaultToSpeaker if hardware refuses direct .playback switch
                try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .allowBluetooth])
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            }

            audioEngine.attach(playerNode)
            audioEngine.connect(
                playerNode,
                to: audioEngine.mainMixerNode,
                format: playbackFormat24kFloatMono
            )
            audioEngine.prepare()
            try audioEngine.start()
        }
    }

    public func configureAndStart() throws {
        try graphQueue.sync {
            resetEngineAndDeactivateSessionLocked()

            let session = AVAudioSession.sharedInstance()
            let mode: AVAudioSession.Mode = (routingMode == .fullDuplexVoiceChatAEC) ? .voiceChat : .spokenAudio

            try session.setCategory(
                .playAndRecord,
                mode: mode,
                options: [.defaultToSpeaker, .allowBluetooth]
            )
            // Request 16kHz hardware hint + 20ms frame duration; AVAudioConverter guarantees
            // exact 16kHz Int16 Mono output even when hardware locks at 48kHz Float32.
            try session.setPreferredSampleRate(targetSampleRate)
            try session.setPreferredIOBufferDuration(0.02)
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            configureAudioGraphLocked()
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
        }
    }

    public func stop() {
        graphQueue.sync {
            resetEngineAndDeactivateSessionLocked()
            ringBuffer.clear()
        }
        resumeAllPlaybackWaiters()
        let active = continuationBox.withLock { current -> AsyncStream<PCM16kChunk>.Continuation? in
            let old = current
            current = nil
            return old
        }
        active?.finish()
    }

    /// Mutes/gates capture when operating in `.halfDuplexGated` mode so Kokoro's audio
    /// cannot bleed back into the ASR recognizer.
    public func setPlaybackCaptureGate(active: Bool) {
        guard routingMode == .halfDuplexGated else {
            isCaptureGatedForPlayback.withLock { $0 = false }
            return
        }
        isCaptureGatedForPlayback.withLock { $0 = active }
    }

    public func cancelPlaybackImmediately() {
        graphQueue.sync {
            playerNode.stop()
            if audioEngine.isRunning, !playerNode.isPlaying {
                playerNode.play()
            }
        }
        setPlaybackCaptureGate(active: false)
        resumeAllPlaybackWaiters()
    }

    /// Enqueues a 24kHz Float32 mono audio buffer onto `AVAudioPlayerNode` for gapless playback.
    /// Applies bounded backpressure (`maxQueuedBuffers`) so Kokoro synthesizes clause N+1
    /// while clause N plays without piling up unbounded PCM buffers in RAM.
    public func enqueuePlayback(samples24k: [Float], maxQueuedBuffers: Int = 2) async {
        guard !samples24k.isEmpty else { return }

        await waitUntilPlaybackCapacityAvailable(maxQueuedBuffers: maxQueuedBuffers)
        if Task.isCancelled { return }

        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: playbackFormat24kFloatMono,
            frameCapacity: AVAudioFrameCount(samples24k.count)
        ) else { return }

        pcmBuffer.frameLength = AVAudioFrameCount(samples24k.count)
        if let channelData = pcmBuffer.floatChannelData?[0] {
            samples24k.withUnsafeBufferPointer { src in
                guard let base = src.baseAddress else { return }
                channelData.update(from: base, count: samples24k.count)
            }
        }

        playbackStateLock.lock()
        pendingPlaybackBuffers += 1
        playbackStateLock.unlock()

        setPlaybackCaptureGate(active: true)
        graphQueue.sync {
            if !audioEngine.isRunning {
                try? audioEngine.start()
            }
            if !playerNode.isPlaying {
                playerNode.play()
            }
            playerNode.scheduleBuffer(pcmBuffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                self?.handleScheduledBufferCompleted()
            }
        }
    }

    /// Suspends until all scheduled `AVAudioPlayerNode` buffers finish playing (or playback is cancelled).
    public func waitForPlaybackDrain() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            playbackStateLock.lock()
            if Task.isCancelled || pendingPlaybackBuffers <= 0 {
                playbackStateLock.unlock()
                continuation.resume()
            } else {
                drainWaiters.append(continuation)
                playbackStateLock.unlock()
            }
        }
    }

    private func waitUntilPlaybackCapacityAvailable(maxQueuedBuffers: Int) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            playbackStateLock.lock()
            if Task.isCancelled || pendingPlaybackBuffers < maxQueuedBuffers {
                playbackStateLock.unlock()
                continuation.resume()
            } else {
                capacityWaiters.append(continuation)
                playbackStateLock.unlock()
            }
        }
    }

    private func handleScheduledBufferCompleted() {
        playbackStateLock.lock()
        pendingPlaybackBuffers = max(0, pendingPlaybackBuffers - 1)
        let capWaiters = capacityWaiters
        capacityWaiters.removeAll()
        let finishedAll = (pendingPlaybackBuffers == 0)
        let drains = finishedAll ? drainWaiters : []
        if finishedAll {
            drainWaiters.removeAll()
        }
        playbackStateLock.unlock()

        capWaiters.forEach { $0.resume() }
        drains.forEach { $0.resume() }
    }

    private func resumeAllPlaybackWaiters() {
        playbackStateLock.lock()
        pendingPlaybackBuffers = 0
        let caps = capacityWaiters
        let drains = drainWaiters
        capacityWaiters.removeAll()
        drainWaiters.removeAll()
        playbackStateLock.unlock()

        caps.forEach { $0.resume() }
        drains.forEach { $0.resume() }
    }

    private func configureAudioGraphLocked() {
        if !audioEngine.attachedNodes.contains(playerNode) {
            audioEngine.attach(playerNode)
            audioEngine.connect(
                playerNode,
                to: audioEngine.mainMixerNode,
                format: playbackFormat24kFloatMono
            )
        }

        let inputNode = audioEngine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)
        guard hwFormat.channelCount > 0, hwFormat.sampleRate > 0 else {
            logger.error("Invalid hardware input format: \(hwFormat)")
            return
        }

        if isTapInstalled {
            inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }

        guard let converter = AVAudioConverter(from: hwFormat, to: captureFormat16kInt16Mono) else {
            logger.error("Unable to create AVAudioConverter from \(hwFormat) to 16kHz Int16 Mono.")
            return
        }
        converter.sampleRateConverterQuality = .max
        self.converter = converter

        // Pre-allocate 0.5s buffer (8,000 samples = 16KB) so CoreAudio HAL tap callbacks
        // up to 4096+ frames never drop audio frames.
        let initialCapacity = AVAudioFrameCount(ceil(0.5 * targetSampleRate))
        self.reusableConvertedBuffer = AVAudioPCMBuffer(
            pcmFormat: captureFormat16kInt16Mono,
            frameCapacity: initialCapacity
        )

        // Tap MUST match `hwFormat` to prevent CoreAudio format mismatch exceptions.
        let hwBufferSize = max(AVAudioFrameCount(1024), AVAudioFrameCount(hwFormat.sampleRate * 0.02))
        inputNode.installTap(onBus: 0, bufferSize: hwBufferSize, format: hwFormat) { [weak self] buffer, when in
            self?.processTapBuffer(buffer, hostTime: when.hostTime)
        }
        isTapInstalled = true
        audioEngine.prepare()
    }

    private func processTapBuffer(_ inputBuffer: AVAudioPCMBuffer, hostTime: UInt64) {
        guard !isCaptureGatedForPlayback.withLock({ $0 }),
              let converter else { return }

        let ratio = targetSampleRate / inputBuffer.format.sampleRate
        let targetFrames = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio)) + 16
        guard targetFrames > 0 else { return }

        if reusableConvertedBuffer == nil || reusableConvertedBuffer!.frameCapacity < targetFrames {
            reusableConvertedBuffer = AVAudioPCMBuffer(
                pcmFormat: captureFormat16kInt16Mono,
                frameCapacity: targetFrames
            )
        }
        guard let outBuffer = reusableConvertedBuffer else { return }

        outBuffer.frameLength = 0
        var error: NSError?
        var inputConsumed = false

        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if inputConsumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            inputConsumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        guard error == nil,
              status == .haveData || status == .inputRanDry,
              outBuffer.frameLength > 0,
              let int16Channels = outBuffer.int16ChannelData
        else { return }

        let frameCount = Int(outBuffer.frameLength)
        let samplesPtr = UnsafeBufferPointer(start: int16Channels[0], count: frameCount)

        ringBuffer.append(samplesPtr)
        let chunkData = Data(bytes: int16Channels[0], count: frameCount * MemoryLayout<Int16>.size)
        let continuation = continuationBox.withLock { $0 }
        continuation?.yield(
            PCM16kChunk(pcmBytes: chunkData, frameCount: frameCount, hostTime: hostTime)
        )
    }

    private func registerAudioNotifications() {
        let center = NotificationCenter.default
        let routeToken = center.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.graphQueue.async {
                let wasRunning = self.audioEngine.isRunning
                self.logger.info("Audio route changed; reconfiguring audio graph.")
                self.configureAudioGraphLocked()
                if wasRunning && !self.audioEngine.isRunning {
                    try? self.audioEngine.start()
                }
            }
        }

        let interruptToken = center.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: nil
        ) { [weak self] notification in
            guard
                let self,
                let info = notification.userInfo,
                let typeVal = info[AVAudioSessionInterruptionTypeKey] as? UInt,
                let type = AVAudioSession.InterruptionType(rawValue: typeVal)
            else { return }

            self.graphQueue.async {
                if type == .began {
                    self.playerNode.pause()
                    self.audioEngine.pause()
                } else if type == .ended {
                    try? AVAudioSession.sharedInstance().setActive(true)
                    try? self.audioEngine.start()
                }
            }
        }
        routeObservers = [routeToken, interruptToken]
    }
}

// MARK: - Deliverable 3A: Sherpa-ONNX ASR Interface & IndicConformer Implementation

public enum ASREvent: Sendable {
    case speechStarted
    case utteranceFinalized(transcript: String, durationSeconds: Double, decodeMs: Double)
}

// MARK: - Bidirectional Indic Script Normalizer & Phonetic Bridge

/// Enforces pure target script fidelity on ASR output (e.g., pure Gurmukhi for Punjabi,
/// pure Devanagari for Hindi without stray Gurmukhi letters, and phonetic Latin rescue for offline English)
/// and provides a native-to-Roman phonetic fallback for Kokoro TTS.
public enum IndicScriptNormalizer {
    private static let punjabiMarkers: [String] = [
        "मैनू", "मैनूं", "मीनू", "तैनू", "तैनूं", "तुसी", "तुसीं", "असीं", "साडा", "साडी", "साडे",
        "तुहाडा", "तुहाडी", "तुहाडे", "किवें", "किद्दां", "की हाल", "वदिया", "वधिया",
        "चंगा", "चंगी", "गल्लां", "गल", "सत श्री", "अकाल", "पंजाब", "करदा", "करदी",
        "होंदा", "होंदी", "नूं", "आं", "दां", "दियां", "वेख", "सुण", "दस", "दसो", "पाजी",
        "हाँजी", "हांजी", "आ गया", "समझ आ", "की कर", "कित्थे", "किधर", "होवे", "सकदे"
    ]

    private static let devaToGurmMulti: [(String, String)] = [
        ("क़", "ਕ"), ("ख़", "ਖ਼"), ("ग़", "ਗ਼"), ("ज़", "ਜ਼"), ("ड़", "ੜ"), ("ढ़", "ੜ੍ਹ"), ("फ़", "ਫ਼"),
        ("क्ष", "ਕਸ਼"), ("त्र", "ਤ੍ਰ"), ("ज्ञ", "ਗ੍ਯ"), ("श्र", "ਸ਼੍ਰ")
    ]

    private static let devaToGurmChar: [Character: String] = [
        "अ": "ਅ", "आ": "ਆ", "इ": "ਇ", "ई": "ਈ", "उ": "ਉ", "ऊ": "ਊ",
        "ए": "ਏ", "ऐ": "ਐ", "ओ": "ਓ", "औ": "ਔ", "ऋ": "ਰਿ",
        "क": "ਕ", "ख": "ਖ", "ग": "ਗ", "घ": "ਘ", "ङ": "ਙ",
        "च": "ਚ", "छ": "ਛ", "ज": "ਜ", "झ": "ਝ", "ञ": "ਞ",
        "ट": "ਟ", "ठ": "ਠ", "ड": "ਡ", "ढ": "ਢ", "ण": "ਣ",
        "त": "ਤ", "थ": "ਥ", "द": "ਦ", "ध": "ਧ", "न": "ਨ",
        "प": "ਪ", "फ": "ਫ", "ब": "ਬ", "भ": "ਭ", "म": "ਮ",
        "य": "ਯ", "र": "ਰ", "ल": "ਲ", "ळ": "ਲ਼", "व": "ਵ",
        "श": "ਸ਼", "ष": "ਸ਼", "स": "ਸ", "ह": "ਹ",
        "ा": "ਾ", "ि": "ਿ", "ी": "ੀ", "ु": "ੁ", "ू": "ੂ",
        "े": "ੇ", "ै": "ੈ", "ो": "ੋ", "ौ": "ੌ",
        "ं": "ੰ", "ँ": "ਂ", "ः": "ਃ", "्": "੍", "़": "",
        "।": "।", "॥": "॥",
        "०": "੦", "१": "੧", "२": "੨", "३": "੩", "४": "੪",
        "५": "੫", "६": "੬", "७": "੭", "८": "੮", "९": "੯"
    ]

    private static let gurmToDevaMulti: [(String, String)] = [
        ("ਖ਼", "ख़"), ("ਗ਼", "ग़"), ("ਜ਼", "ज़"), ("ਫ਼", "फ़"), ("ਲ਼", "ळ"), ("ਸ਼", "श"), ("ੜ੍ਹ", "ढ़")
    ]

    private static let gurmToDevaChar: [Character: String] = [
        "ਅ": "अ", "ਆ": "आ", "ਇ": "इ", "ਈ": "ई", "ਉ": "उ", "ਊ": "ऊ",
        "ਏ": "ए", "ਐ": "ऐ", "ਓ": "ओ", "ਔ": "औ",
        "ਕ": "क", "ਖ": "ख", "ਗ": "ग", "ਘ": "घ", "ਙ": "ङ",
        "ਚ": "च", "ਛ": "छ", "ਜ": "ज", "ਝ": "झ", "ਞ": "ञ",
        "ਟ": "ट", "ਠ": "ठ", "ਡ": "ड", "ਢ": "ढ", "ਣ": "ण",
        "ਤ": "त", "ਥ": "थ", "ਦ": "द", "ਧ": "ध", "ਨ": "न",
        "ਪ": "प", "ਫ": "फ", "ਬ": "ब", "ਭ": "भ", "ਮ": "म",
        "ਯ": "य", "ਰ": "र", "ਲ": "ल", "ਵ": "व", "ਸ": "स", "ਹ": "ह", "ੜ": "ड़",
        "ਾ": "ा", "ਿ": "ि", "ੀ": "ी", "ੁ": "ु", "ੂ": "ू",
        "ੇ": "े", "ੈ": "ै", "ੋ": "ो", "ੌ": "ौ",
        "ੰ": "ं", "ਂ": "ं", "ੱ": "", "ਃ": "ः", "੍": "्", "਼": "",
        "੦": "०", "੧": "१", "੨": "२", "੩": "३", "੪": "४",
        "੫": "५", "੬": "६", "੭": "७", "੮": "८", "੯": "९"
    ]

    public static func normalize(
        _ text: String,
        inputLanguage: PipelineLanguage,
        legacyMode: VoicePipelineConfiguration.ScriptDisplayMode = .auto
    ) -> String {
        switch inputLanguage {
        case .punjabi:
            return convertDevanagariToGurmukhi(text)
        case .hindi, .marathi:
            return convertGurmukhiToDevanagari(text)
        case .english:
            let hasIndic = text.unicodeScalars.contains {
                ($0.value >= 0x0900 && $0.value <= 0x097F) || ($0.value >= 0x0A00 && $0.value <= 0x0A7F)
            }
            return hasIndic ? toPhoneticLatin(text) : text
        case .auto:
            guard legacyMode != .nativeASR else { return text }
            if legacyMode == .punjabiGurmukhi {
                return convertDevanagariToGurmukhi(text)
            }
            let hasGurmukhi = text.unicodeScalars.contains { $0.value >= 0x0A00 && $0.value <= 0x0A7F }
            let isPunjabi = hasGurmukhi || punjabiMarkers.contains { text.contains($0) }
            if isPunjabi {
                return convertDevanagariToGurmukhi(text)
            }
            return convertGurmukhiToDevanagari(text)
        default:
            return text
        }
    }

    public static func convertDevanagariToGurmukhi(_ text: String) -> String {
        let containsDeva = text.unicodeScalars.contains { $0.value >= 0x0900 && $0.value <= 0x097F }
        guard containsDeva else { return text }

        var working = text
        for (src, dst) in devaToGurmMulti {
            working = working.replacingOccurrences(of: src, with: dst)
        }
        var out = String()
        out.reserveCapacity(working.count)
        for ch in working {
            out.append(devaToGurmChar[ch] ?? String(ch))
        }
        let polish: [(String, String)] = [
            ("ਮੈਨੂ", "ਮੈਨੂੰ"), ("ਮੀਨੂ", "ਮੈਨੂੰ"), ("ਤੈਨੂ", "ਤੈਨੂੰ"),
            ("ਤੁਸੀ", "ਤੁਸੀਂ"), ("ਅਸੀ", "ਅਸੀਂ"), ("ਨਹੀ ", "ਨਹੀਂ "),
            ("ਹਾਂ ਜੀ", "ਹਾਂਜੀ"), ("ਸਤ ਸ੍ਰੀ", "ਸਤਿ ਸ੍ਰੀ")
        ]
        for (k, v) in polish {
            out = out.replacingOccurrences(of: k, with: v)
        }
        return out
    }

    public static func convertGurmukhiToDevanagari(_ text: String) -> String {
        let containsGurm = text.unicodeScalars.contains { $0.value >= 0x0A00 && $0.value <= 0x0A7F }
        guard containsGurm else { return text }

        var working = text
        for (src, dst) in gurmToDevaMulti {
            working = working.replacingOccurrences(of: src, with: dst)
        }
        var out = String()
        out.reserveCapacity(working.count)
        for ch in working {
            out.append(gurmToDevaChar[ch] ?? String(ch))
        }
        return out
    }

    /// Converts Gurmukhi or Devanagari characters into speakable Latin phonetics.
    public static func toPhoneticLatin(_ text: String) -> String {
        let deva = convertGurmukhiToDevanagari(text)
        let consonants: [Character: String] = [
            "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
            "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
            "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
            "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
            "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
            "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
            "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ड़": "r"
        ]
        let indepVowels: [Character: String] = [
            "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
            "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au"
        ]
        let matras: [Character: String] = [
            "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
            "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "्": ""
        ]
        let scalars = Array(deva)
        var out = ""
        var i = 0
        while i < scalars.count {
            let ch = scalars[i]
            if let c = consonants[ch] {
                out.append(c)
                if i + 1 < scalars.count, let m = matras[scalars[i + 1]] {
                    out.append(m)
                    i += 2
                } else {
                    if i + 1 < scalars.count && scalars[i + 1] != " " {
                        out.append("a")
                    }
                    i += 1
                }
            } else if let v = indepVowels[ch] {
                out.append(v)
                i += 1
            } else if ch == "ं" || ch == "ँ" {
                out.append("n")
                i += 1
            } else if ch == "।" {
                out.append(". ")
                i += 1
            } else if ch == "़" {
                i += 1
            } else {
                out.append(ch)
                i += 1
            }
        }
        return out
    }
}

public protocol ASREngineProtocol: Actor {
    var isLoaded: Bool { get }
    func initialize() async throws
    func updateInputLanguage(_ language: PipelineLanguage, legacyMode: VoicePipelineConfiguration.ScriptDisplayMode)
    func processPCMChunk(_ chunk: PCM16kChunk) async throws -> [ASREvent]
    func resetVAD()
    func unloadModel() async
}

public actor SherpaIndicConformerASR: ASREngineProtocol {
    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "SherpaASR")
    private var config: VoicePipelineConfiguration.ASRConfig

    public private(set) var isLoaded: Bool = false
    private var floatScratchBuffer: [Float] = []
    private var isCurrentlyInSpeech: Bool = false

    #if canImport(Speech)
    private let englishSpeechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-IN"))
        ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    #endif

    #if canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED
    private var recognizer: SherpaOnnxOfflineRecognizer?
    private var vad: SherpaOnnxVoiceActivityDetectorWrapper?
    #else
    // Structural simulation accumulators when SherpaOnnx XCFramework is not yet linked.
    private var mockAccumulatedFrames: Int = 0
    private var mockSilenceFrames: Int = 0
    #endif

    public init(config: VoicePipelineConfiguration.ASRConfig) {
        self.config = config
    }

    public func updateInputLanguage(_ language: PipelineLanguage, legacyMode: VoicePipelineConfiguration.ScriptDisplayMode) {
        self.config = VoicePipelineConfiguration.ASRConfig(
            indicConformerModelPath: config.indicConformerModelPath,
            tokensPath: config.tokensPath,
            sileroVadModelPath: config.sileroVadModelPath,
            sampleRate: config.sampleRate,
            numThreads: config.numThreads,
            provider: config.provider,
            vadThreshold: config.vadThreshold,
            minSilenceDurationSeconds: config.minSilenceDurationSeconds,
            minSpeechDurationSeconds: config.minSpeechDurationSeconds,
            maxUtteranceSeconds: config.maxUtteranceSeconds,
            scriptDisplayMode: legacyMode,
            inputLanguage: language
        )
    }

    private func recognizeEnglishIfAuthorized(samples: [Float], sampleRate: Double) async -> String? {
        #if canImport(Speech)
        guard SFSpeechRecognizer.authorizationStatus() == .authorized,
              let speechRec = englishSpeechRecognizer,
              speechRec.isAvailable else {
            return nil
        }
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let dst = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    dst.initialize(from: base, count: samples.count)
                }
            }
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        if speechRec.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        request.shouldReportPartialResults = false
        request.append(buffer)
        request.endAudio()

        return await withCheckedContinuation { continuation in
            var resumed = false
            speechRec.recognitionTask(with: request) { result, error in
                if let result, result.isFinal, !resumed {
                    resumed = true
                    continuation.resume(returning: result.bestTranscription.formattedString)
                } else if error != nil, !resumed {
                    resumed = true
                    continuation.resume(returning: nil)
                }
            }
        }
        #else
        return nil
        #endif
    }

    public func initialize() async throws {
        guard !isLoaded else { return }

        #if canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED
        for path in [config.indicConformerModelPath, config.tokensPath, config.sileroVadModelPath] {
            guard FileManager.default.fileExists(atPath: path) else {
                throw VoicePipelineError.modelFileMissing(path)
            }
        }

        // 1. Configure Silero VAD (16kHz, 512-sample window = 32ms)
        let sileroConfig = sherpaOnnxSileroVadModelConfig(
            model: config.sileroVadModelPath,
            threshold: config.vadThreshold,
            minSilenceDuration: config.minSilenceDurationSeconds,
            minSpeechDuration: config.minSpeechDurationSeconds,
            windowSize: 512
        )
        var vadConfig = sherpaOnnxVadModelConfig(
            sileroVad: sileroConfig,
            sampleRate: Int32(config.sampleRate),
            numThreads: 1,
            provider: "cpu",
            debug: 0
        )
        self.vad = SherpaOnnxVoiceActivityDetectorWrapper(
            config: &vadConfig,
            buffer_size_in_seconds: Float(config.maxUtteranceSeconds)
        )

        // 2. Configure IndicConformer INT8 ONNX via NeMo EncDec CTC config
        let nemoCtcConfig = sherpaOnnxOfflineNemoEncDecCtcModelConfig(
            model: config.indicConformerModelPath
        )
        let offlineModelConfig = sherpaOnnxOfflineModelConfig(
            tokens: config.tokensPath,
            nemoCtc: nemoCtcConfig,
            numThreads: Int(config.numThreads),
            provider: config.provider,
            debug: 0
        )
        let featConfig = sherpaOnnxFeatureConfig(sampleRate: Int(config.sampleRate), featureDim: 80)
        var recognizerConfig = sherpaOnnxOfflineRecognizerConfig(
            featConfig: featConfig,
            modelConfig: offlineModelConfig,
            decodingMethod: "greedy_search",
            maxActivePaths: 4
        )
        self.recognizer = SherpaOnnxOfflineRecognizer(config: &recognizerConfig)
        #else
        logger.info("Initialized structural mock SherpaIndicConformerASR (link SherpaOnnx XCFramework for live inference).")
        #endif

        isLoaded = true
    }

    public func processPCMChunk(_ chunk: PCM16kChunk) async throws -> [ASREvent] {
        guard isLoaded else {
            throw VoicePipelineError.engineNotInitialized("SherpaIndicConformerASR")
        }

        // Reuse floatScratchBuffer across callbacks to avoid GC/ARC churn.
        chunk.copyNormalizedFloat32(into: &floatScratchBuffer)
        var events: [ASREvent] = []

        #if canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED
        guard let vad, let recognizer else { return events }
        vad.acceptWaveform(samples: floatScratchBuffer)

        let speechNow = vad.isSpeechDetected()
        if speechNow && !isCurrentlyInSpeech {
            isCurrentlyInSpeech = true
            events.append(.speechStarted)
        }

        while !vad.isEmpty() {
            let segment = vad.front()
            let samples = segment.samples
            vad.pop()
            isCurrentlyInSpeech = false

            guard !samples.isEmpty else { continue }
            let maxSamples = Int(Float(config.sampleRate) * Float(config.maxUtteranceSeconds))
            let boundedSamples = samples.count > maxSamples ? Array(samples.prefix(maxSamples)) : samples

            let decodeStart = CFAbsoluteTimeGetCurrent()
            var finalText = ""
            if config.inputLanguage == .english,
               let appleEnglish = await recognizeEnglishIfAuthorized(samples: boundedSamples, sampleRate: Double(config.sampleRate)),
               !appleEnglish.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
                finalText = appleEnglish.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            } else {
                let result = recognizer.decode(samples: boundedSamples, sampleRate: Int(config.sampleRate))
                let rawText = result.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                if !rawText.isEmpty {
                    finalText = IndicScriptNormalizer.normalize(
                        rawText,
                        inputLanguage: config.inputLanguage,
                        legacyMode: config.scriptDisplayMode
                    )
                }
            }
            let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000.0
            if !finalText.isEmpty {
                let duration = Double(boundedSamples.count) / Double(config.sampleRate)
                events.append(.utteranceFinalized(transcript: finalText, durationSeconds: duration, decodeMs: decodeMs))
            }
        }
        #else
        // Structural energy-based mock VAD + sample Gurmukhi utterance trigger for integration testing.
        var rms: Float = 0
        vDSP_rmsqv(floatScratchBuffer, 1, &rms, vDSP_Length(floatScratchBuffer.count))
        if rms > 0.015 {
            if !isCurrentlyInSpeech {
                isCurrentlyInSpeech = true
                events.append(.speechStarted)
            }
            mockAccumulatedFrames += chunk.frameCount
            mockSilenceFrames = 0
        } else if isCurrentlyInSpeech {
            mockSilenceFrames += chunk.frameCount
            if mockSilenceFrames >= Int(Float(config.sampleRate) * config.minSilenceDurationSeconds) {
                isCurrentlyInSpeech = false
                let duration = Double(mockAccumulatedFrames) / Double(config.sampleRate)
                mockAccumulatedFrames = 0
                mockSilenceFrames = 0
                if duration >= Double(config.minSpeechDurationSeconds) {
                    events.append(
                        .utteranceFinalized(
                            transcript: "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
                            durationSeconds: duration,
                            decodeMs: 45.0
                        )
                    )
                }
            }
        }
        #endif

        return events
    }

    public func resetVAD() {
        isCurrentlyInSpeech = false
        #if canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED
        vad?.clear()
        #else
        mockAccumulatedFrames = 0
        mockSilenceFrames = 0
        #endif
    }

    public func unloadModel() async {
        #if canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED
        recognizer = nil
        vad = nil
        #endif
        floatScratchBuffer.removeAll(keepingCapacity: false)
        isCurrentlyInSpeech = false
        isLoaded = false
        logger.info("Unloaded Sherpa-ONNX ASR & VAD models to reclaim RAM.")
    }
}

// MARK: - Deliverable 3B: LiteRT-LM Gemma 4 E2B-IT Interface & Implementation

public protocol LLMBrainProtocol: Actor {
    var isLoaded: Bool { get }
    func loadEngine() async throws
    func updateSystemPrompt(_ newPrompt: String) async throws
    func streamResponse(for userTranscript: String) -> AsyncThrowingStream<String, Error>
    func resetConversation() async throws
    func unloadEngine() async
}

public actor LiteRTLMGemmaBrain: LLMBrainProtocol {
    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "LiteRTLMGemma")
    private var config: VoicePipelineConfiguration.LLMConfig

    public private(set) var isLoaded: Bool = false
    private var turnCount: Int = 0

    #if canImport(LiteRTLM)
    private var engine: Engine?
    private var conversation: Conversation?
    #endif

    public init(config: VoicePipelineConfiguration.LLMConfig) {
        self.config = config
    }

    public func loadEngine() async throws {
        guard !isLoaded else { return }

        #if canImport(LiteRTLM)
        guard FileManager.default.fileExists(atPath: config.modelPath) else {
            throw VoicePipelineError.modelFileMissing(config.modelPath)
        }

        let engineConfig = try EngineConfig(
            modelPath: config.modelPath,
            backend: .gpu, // Metal GPU hardware acceleration
            maxNumTokens: config.maxContextTokens,
            cacheDir: config.cacheDirectory
        )
        let newEngine = Engine(engineConfig: engineConfig)
        try await newEngine.initialize()
        self.engine = newEngine
        try await createFreshConversation()
        #else
        logger.info("Initialized structural mock LiteRTLMGemmaBrain (link LiteRTLM SPM package for Metal GPU Gemma 4).")
        #endif

        isLoaded = true
    }

    public func updateSystemPrompt(_ newPrompt: String) async throws {
        self.config = VoicePipelineConfiguration.LLMConfig(
            modelPath: config.modelPath,
            cacheDirectory: config.cacheDirectory,
            maxContextTokens: config.maxContextTokens,
            topK: config.topK,
            topP: config.topP,
            temperature: config.temperature,
            maxTurnsBeforeKVReset: config.maxTurnsBeforeKVReset,
            systemPrompt: newPrompt
        )
        turnCount = 0
        #if canImport(LiteRTLM)
        if engine != nil {
            try await createFreshConversation()
        }
        #endif
    }

    public func resetConversation() async throws {
        turnCount = 0
        #if canImport(LiteRTLM)
        try await createFreshConversation()
        #endif
    }

    public func streamResponse(for userTranscript: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    if !self.isLoaded {
                        try await self.loadEngine()
                    }
                    // Periodically reset conversation KV cache so long multi-turn sessions
                    // never exhaust iOS unified GPU/CPU memory.
                    if self.turnCount >= self.config.maxTurnsBeforeKVReset {
                        self.logger.info("Resetting Gemma 4 KV cache after \(self.turnCount) turns.")
                        try await self.resetConversation()
                    }
                    self.turnCount += 1

                    #if canImport(LiteRTLM)
                    guard let conversation = self.conversation else {
                        throw VoicePipelineError.engineNotInitialized("LiteRTLM Conversation")
                    }
                    let userMessage = Message(userTranscript)
                    for try await chunk in conversation.sendMessageStream(userMessage) {
                        try Task.checkCancellation()
                        let tokenText = chunk.toString
                        if !tokenText.isEmpty {
                            continuation.yield(tokenText)
                        }
                    }
                    #else
                    // Structural transliterated response stream matching the system prompt constraints.
                    let simulatedTokens = [
                        "Sat ", "sri ", "akaal! ",
                        "Main ", "bilkul ", "vadiya ", "aan. ",
                        "Tusi ", "daso, ", "aaj ", "main ", "tuhadi ", "ki ", "madad ", "kara?"
                    ]
                    for token in simulatedTokens {
                        try Task.checkCancellation()
                        try await Task.sleep(nanoseconds: 25_000_000)
                        continuation.yield(token)
                    }
                    #endif
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func unloadEngine() async {
        #if canImport(LiteRTLM)
        conversation = nil
        engine = nil
        #endif
        turnCount = 0
        isLoaded = false
        logger.info("Unloaded LiteRT-LM Gemma 4 Metal engine.")
    }

    #if canImport(LiteRTLM)
    private func createFreshConversation() async throws {
        guard let engine else {
            throw VoicePipelineError.engineNotInitialized("LiteRTLM Engine")
        }
        let samplerConfig = try SamplerConfig(
            topK: config.topK,
            topP: Float(config.topP),
            temperature: Float(config.temperature)
        )
        let conversationConfig = ConversationConfig(
            systemMessage: Message(config.systemPrompt),
            samplerConfig: samplerConfig
        )
        self.conversation = try await engine.createConversation(with: conversationConfig)
    }
    #endif
}

// MARK: - Linguistic Bridge: Native-Script Sanitizer & Low-Latency Voice Clause Chunker

/// Guarantees Gemma 4's output obeys the Native Script Ban and streams complete
/// speakable clauses to Kokoro-TTS before full LLM generation finishes.
public struct TransliterationStreamChunker: Sendable {
    private var buffer: String = ""
    private let minClauseLengthForCommaSplit = 42
    private let maxClauseLength = 210

    public init() {}

    public mutating func append(token: String) -> [String] {
        let sanitized = Self.sanitizeForTTS(token)
        guard !sanitized.isEmpty else { return [] }
        buffer.append(sanitized)

        var readyClauses: [String] = []
        while let splitIndex = findClauseBoundary(in: buffer) {
            let rawClause = String(buffer[..<splitIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            buffer.removeSubrange(..<splitIndex)
            if !rawClause.isEmpty {
                readyClauses.append(rawClause)
            }
        }
        return readyClauses
    }

    public mutating func flush() -> String? {
        let remaining = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer.removeAll(keepingCapacity: false)
        return remaining.isEmpty ? nil : remaining
    }

    private func findClauseBoundary(in text: String) -> String.Index? {
        let sentenceTerminators: Set<Character> = [".", "!", "?"]
        for idx in text.indices {
            let ch = text[idx]
            if sentenceTerminators.contains(ch) {
                return text.index(after: idx)
            }
            if (ch == "," || ch == ";" || ch == ":"),
               text.distance(from: text.startIndex, to: idx) >= minClauseLengthForCommaSplit {
                return text.index(after: idx)
            }
        }
        if text.count >= maxClauseLength,
           let lastSpace = text.lastIndex(of: " ") {
            return text.index(after: lastSpace)
        }
        return nil
    }

    /// Strips markdown symbols and filters out any leaked non-Latin Indic script scalar
    /// (U+0900..U+0D7F) so Kokoro's G2P tokenizer never receives unsupported glyphs.
    public static func sanitizeForTTS(_ raw: String) -> String {
        var result = String()
        result.reserveCapacity(raw.count)

        for scalar in raw.unicodeScalars {
            // Strip common markdown/formatting characters disallowed by voice constraints.
            if scalar == "*" || scalar == "`" || scalar == "#" || scalar == "_" {
                continue
            }
            // Enforce Native Script Ban across Indic Unicode blocks (Devanagari..Malayalam).
            if (0x0900...0x0D7F).contains(scalar.value) {
                continue
            }
            result.unicodeScalars.append(scalar)
        }
        return result
    }
}

// MARK: - Deliverable 3C: Kokoro 82M ONNX Runtime Interface, Voice Pack & Phonemizer

public protocol KokoroPhonemizerProtocol: Sendable {
    /// Converts clean Romanised text into Kokoro vocabulary IDs (excluding start/end 0 padding).
    /// Must never exceed `maxTokens` (510).
    func phonemizeAndTokenize(_ text: String, maxTokens: Int) throws -> [Int64]
}

/// Structural phonemizer translating Romanised Indian/English prose into IPA token IDs
/// loaded from Kokoro's `vocab.json`. Replace/back `espeak-ng` or Misaki C-bridge for
/// production phoneme accuracy.
public struct IndicRomanKokoroPhonemizer: KokoroPhonemizerProtocol {
    private let vocab: [String: Int64]

    public init(vocabJSONPath: String) throws {
        if FileManager.default.fileExists(atPath: vocabJSONPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: vocabJSONPath))
            let decoded = try JSONDecoder().decode([String: Int64].self, from: data)
            self.vocab = decoded
        } else {
            // Structural fallback vocabulary subset for mock/dry-run validation.
            self.vocab = [" ": 16, "a": 43, "e": 47, "i": 51, "o": 57, "u": 63,
                          "k": 53, "t": 62, "s": 61, "n": 56, "m": 55, "v": 64,
                          "d": 46, "y": 67, "l": 54, "r": 60, "h": 50, "p": 58,
                          ".": 4, "!": 5, "?": 6, ",": 3]
        }
    }

    public func phonemizeAndTokenize(_ text: String, maxTokens: Int = 510) throws -> [Int64] {
        // Normalize Indian phonetic digraphs toward IPA equivalents before vocab lookup.
        let normalized = text.lowercased()
            .replacingOccurrences(of: "aa", with: "ɑː")
            .replacingOccurrences(of: "ee", with: "iː")
            .replacingOccurrences(of: "oo", with: "uː")
            .replacingOccurrences(of: "sh", with: "ʃ")
            .replacingOccurrences(of: "ch", with: "tʃ")
            .replacingOccurrences(of: "ng", with: "ŋ")

        var ids: [Int64] = []
        ids.reserveCapacity(min(normalized.count, maxTokens))

        for ch in normalized {
            let key = String(ch)
            if let id = vocab[key] {
                ids.append(id)
            } else if let asciiFallback = key.folding(options: .diacriticInsensitive, locale: .current).first,
                      let fallbackID = vocab[String(asciiFallback)] {
                ids.append(fallbackID)
            }
            if ids.count >= maxTokens { break }
        }

        guard !ids.isEmpty else {
            throw VoicePipelineError.phonemizationFailed(text)
        }
        return ids
    }
}

/// Memory-mapped style embedding pack reader for Kokoro-82M (`[N, 1, 256]` float32 `.bin`).
/// Using `.mappedIfSafe` keeps unused style slices paged out on disk instead of in dirty heap RAM.
public final class KokoroVoicePack: @unchecked Sendable {
    private let mappedData: Data
    private let styleDim = 256
    private let frameByteStride = 256 * MemoryLayout<Float>.size
    public let frameCount: Int

    public init(voicePackPath: String) throws {
        guard FileManager.default.fileExists(atPath: voicePackPath) else {
            throw VoicePipelineError.modelFileMissing(voicePackPath)
        }
        let url = URL(fileURLWithPath: voicePackPath)
        self.mappedData = try Data(contentsOf: url, options: .mappedIfSafe)
        self.frameCount = mappedData.count / frameByteStride
        guard frameCount > 0 else {
            throw VoicePipelineError.invalidTensorShape("Voice pack at \(voicePackPath) has 0 frames.")
        }
    }

    /// Kokoro indexes style vectors by unpadded token length (`voices[len(tokens)]`),
    /// clamped to `frameCount - 1`.
    public func styleVector(forTokenCount tokenCount: Int) -> [Float] {
        let clampedIndex = min(max(0, tokenCount), frameCount - 1)
        let offset = clampedIndex * frameByteStride
        var vector = [Float](repeating: 0, count: styleDim)
        _ = vector.withUnsafeMutableBytes { dst in
            mappedData.copyBytes(to: dst, from: offset..<(offset + frameByteStride))
        }
        return vector
    }
}

public protocol TTSEngineProtocol: Actor {
    var isLoaded: Bool { get }
    func loadSession() async throws
    func synthesize(phoneticClause: String, speed: Float) async throws -> [Float]
    func unloadSession() async
}

public actor KokoroONNXSynthesizer: TTSEngineProtocol {
    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "KokoroTTS")
    private let config: VoicePipelineConfiguration.TTSConfig
    private let phonemizer: KokoroPhonemizerProtocol

    public private(set) var isLoaded: Bool = false
    private var voicePack: KokoroVoicePack?

    #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc)
    private var ortEnv: ORTEnv?
    private var ortSession: ORTSession?
    private var cachedInputNames: Set<String> = []
    private var cachedOutputName: String = "waveform"
    #elseif canImport(onnxruntime)
    private var ortApi: UnsafePointer<OrtApi>?
    private var ortEnv: OpaquePointer?
    private var ortSession: OpaquePointer?
    private var cachedInputNames: Set<String> = []
    private var cachedOutputName: String = "waveform"
    #endif

    public init(
        config: VoicePipelineConfiguration.TTSConfig,
        phonemizer: KokoroPhonemizerProtocol? = nil
    ) throws {
        self.config = config
        self.phonemizer = try phonemizer ?? IndicRomanKokoroPhonemizer(vocabJSONPath: config.vocabJSONPath)
    }

    public func loadSession() async throws {
        guard !isLoaded else { return }

        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc)
        guard FileManager.default.fileExists(atPath: config.kokoroModelPath) else {
            throw VoicePipelineError.modelFileMissing(config.kokoroModelPath)
        }
        self.voicePack = try KokoroVoicePack(voicePackPath: config.voicePackPath)

        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(config.intraOpThreads)
        try options.setGraphOptimizationLevel(.all)
        // Share environment-level allocators & avoid duplicating weights in memory arenas.
        try options.addConfigEntry(withKey: "session.use_env_allocators", value: "1")

        if config.useCoreMLExecutionProvider {
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            try options.appendCoreMLExecutionProvider(with: coreMLOptions)
        }

        let session = try ORTSession(env: env, modelPath: config.kokoroModelPath, sessionOptions: options)
        self.ortEnv = env
        self.ortSession = session
        self.cachedInputNames = Set(try session.inputNames())
        if let firstOutput = try session.outputNames().first {
            self.cachedOutputName = firstOutput
        }
        #elseif canImport(onnxruntime)
        guard FileManager.default.fileExists(atPath: config.kokoroModelPath) else {
            throw VoicePipelineError.modelFileMissing(config.kokoroModelPath)
        }
        self.voicePack = try KokoroVoicePack(voicePackPath: config.voicePackPath)

        guard let apiBase = OrtGetApiBase(),
              let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw VoicePipelineError.engineNotInitialized("Failed to initialize ONNX Runtime C API")
        }
        self.ortApi = api

        var env: OpaquePointer?
        try checkORTStatus(api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "KokoroTTS", &env), api: api)

        var opts: OpaquePointer?
        try checkORTStatus(api.pointee.CreateSessionOptions(&opts), api: api)
        defer { if let opts { api.pointee.ReleaseSessionOptions(opts) } }

        try checkORTStatus(api.pointee.SetIntraOpNumThreads(opts, Int32(config.intraOpThreads)), api: api)
        try checkORTStatus(api.pointee.SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL), api: api)

        var session: OpaquePointer?
        try config.kokoroModelPath.withCString { pathPtr in
            try checkORTStatus(api.pointee.CreateSession(env, pathPtr, opts, &session), api: api)
        }

        self.ortEnv = env
        self.ortSession = session

        var allocator: UnsafeMutablePointer<OrtAllocator>?
        try checkORTStatus(api.pointee.GetAllocatorWithDefaultOptions(&allocator), api: api)
        var inputCount: Int = 0
        try checkORTStatus(api.pointee.SessionGetInputCount(session, &inputCount), api: api)
        var names = Set<String>()
        for i in 0..<inputCount {
            var namePtr: UnsafeMutablePointer<CChar>?
            try checkORTStatus(api.pointee.SessionGetInputName(session, i, allocator, &namePtr), api: api)
            if let namePtr {
                names.insert(String(cString: namePtr))
                allocator?.pointee.Free(allocator, namePtr)
            }
        }
        self.cachedInputNames = names

        var outputCount: Int = 0
        try checkORTStatus(api.pointee.SessionGetOutputCount(session, &outputCount), api: api)
        if outputCount > 0 {
            var outNamePtr: UnsafeMutablePointer<CChar>?
            try checkORTStatus(api.pointee.SessionGetOutputName(session, 0, allocator, &outNamePtr), api: api)
            if let outNamePtr {
                self.cachedOutputName = String(cString: outNamePtr)
                allocator?.pointee.Free(allocator, outNamePtr)
            }
        }
        #else
        logger.info("Initialized structural mock KokoroONNXSynthesizer (link OnnxRuntimeBindings SPM package for live synthesis).")
        #endif

        isLoaded = true
    }

    public func synthesize(phoneticClause: String, speed: Float) async throws -> [Float] {
        if !isLoaded {
            try await loadSession()
        }

        let rawTokens = try phonemizer.phonemizeAndTokenize(
            phoneticClause,
            maxTokens: config.maxTokenLength
        )

        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc)
        guard let ortSession, let voicePack else {
            throw VoicePipelineError.engineNotInitialized("Kokoro ORTSession")
        }

        // Kokoro expects padding token `0` at start and end: [0, t1, ..., tn, 0] (max <= 512).
        var paddedInputIDs: [Int64] = [0]
        paddedInputIDs.reserveCapacity(rawTokens.count + 2)
        paddedInputIDs.append(contentsOf: rawTokens)
        paddedInputIDs.append(0)

        let styleVector = voicePack.styleVector(forTokenCount: rawTokens.count)
        let speedScalar: [Float] = [speed]

        // CRITICAL iOS MEMORY GUARD: Execute every ORTValue creation & session.run inside an
        // `autoreleasepool` so bridged NSMutableData / ORTValue tensors release immediately
        // after extracting the 24kHz Float32 waveform slice.
        return try autoreleasepool {
            let inputIDsData = paddedInputIDs.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Int64>.size) }
            let styleData = styleVector.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Float>.size) }
            let speedData = speedScalar.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Float>.size) }

            let inputIDsValue = try ORTValue(
                tensorData: inputIDsData,
                elementType: .int64,
                shape: [1, NSNumber(value: paddedInputIDs.count)]
            )
            let styleValue = try ORTValue(
                tensorData: styleData,
                elementType: .float,
                shape: [1, 256]
            )
            let speedValue = try ORTValue(
                tensorData: speedData,
                elementType: .float,
                shape: [1]
            )

            // Supports both Kokoro-82M-v1.0-ONNX (`input_ids`) and v0.19 (`tokens`) graph names.
            let tokenInputKey = cachedInputNames.contains("input_ids") ? "input_ids" : "tokens"
            let outputs = try ortSession.run(
                withInputs: [
                    tokenInputKey: inputIDsValue,
                    "style": styleValue,
                    "speed": speedValue
                ],
                outputNames: [cachedOutputName],
                runOptions: nil
            )

            guard let waveformTensor = outputs[cachedOutputName] else {
                throw VoicePipelineError.invalidTensorShape("Missing output tensor '\(cachedOutputName)'.")
            }

            let rawWaveformData = try waveformTensor.tensorData() as Data
            let sampleCount = rawWaveformData.count / MemoryLayout<Float>.size
            var samples = [Float](repeating: 0, count: sampleCount)
            _ = samples.withUnsafeMutableBytes { dst in
                rawWaveformData.copyBytes(to: dst)
            }
            return samples
        }
        #elseif canImport(onnxruntime)
        guard let api = ortApi, let ortSession, let voicePack else {
            throw VoicePipelineError.engineNotInitialized("Kokoro ORT C Session")
        }

        var paddedInputIDs: [Int64] = [0]
        paddedInputIDs.reserveCapacity(rawTokens.count + 2)
        paddedInputIDs.append(contentsOf: rawTokens)
        paddedInputIDs.append(0)

        var styleVector = voicePack.styleVector(forTokenCount: rawTokens.count)
        var speedScalar: [Float] = [speed]

        return try autoreleasepool {
            var memInfo: OpaquePointer?
            try checkORTStatus(api.pointee.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo), api: api)
            defer { if let memInfo { api.pointee.ReleaseMemoryInfo(memInfo) } }

            var inputIDsValue: OpaquePointer?
            var styleValue: OpaquePointer?
            var speedValue: OpaquePointer?
            defer {
                if let inputIDsValue { api.pointee.ReleaseValue(inputIDsValue) }
                if let styleValue { api.pointee.ReleaseValue(styleValue) }
                if let speedValue { api.pointee.ReleaseValue(speedValue) }
            }

            var idShape: [Int64] = [1, Int64(paddedInputIDs.count)]
            var styleShape: [Int64] = [1, 256]
            var speedShape: [Int64] = [1]

            let tokenInputKey = cachedInputNames.contains("input_ids") ? "input_ids" : "tokens"
            let cInputNames: [UnsafePointer<CChar>?] = [
                UnsafePointer(strdup(tokenInputKey)),
                UnsafePointer(strdup("style")),
                UnsafePointer(strdup("speed"))
            ]
            defer { cInputNames.forEach { if let p = $0 { free(UnsafeMutablePointer(mutating: p)) } } }

            let cOutputName = UnsafePointer(strdup(cachedOutputName))
            defer { if let p = cOutputName { free(UnsafeMutablePointer(mutating: p)) } }

            var outputValues: [OpaquePointer?] = [nil]

            try paddedInputIDs.withUnsafeMutableBytes { idBuf in
                try styleVector.withUnsafeMutableBytes { styleBuf in
                    try speedScalar.withUnsafeMutableBytes { speedBuf in
                        try checkORTStatus(api.pointee.CreateTensorWithDataAsOrtValue(
                            memInfo, idBuf.baseAddress, idBuf.count, &idShape, 2,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputIDsValue
                        ), api: api)
                        try checkORTStatus(api.pointee.CreateTensorWithDataAsOrtValue(
                            memInfo, styleBuf.baseAddress, styleBuf.count, &styleShape, 2,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &styleValue
                        ), api: api)
                        try checkORTStatus(api.pointee.CreateTensorWithDataAsOrtValue(
                            memInfo, speedBuf.baseAddress, speedBuf.count, &speedShape, 1,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &speedValue
                        ), api: api)

                        var inputs: [OpaquePointer?] = [inputIDsValue, styleValue, speedValue]
                        var outputNames: [UnsafePointer<CChar>?] = [cOutputName]

                        try checkORTStatus(api.pointee.Run(
                            ortSession, nil,
                            cInputNames, &inputs, 3,
                            outputNames, 1, &outputValues
                        ), api: api)
                    }
                }
            }

            guard let outVal = outputValues[0] else {
                throw VoicePipelineError.invalidTensorShape("Missing output tensor '\(cachedOutputName)'.")
            }
            defer { api.pointee.ReleaseValue(outVal) }

            var typeInfo: OpaquePointer?
            try checkORTStatus(api.pointee.GetTensorTypeAndShape(outVal, &typeInfo), api: api)
            defer { if let typeInfo { api.pointee.ReleaseTensorTypeAndShapeInfo(typeInfo) } }

            var elemCount: Int = 0
            try checkORTStatus(api.pointee.GetTensorShapeElementCount(typeInfo, &elemCount), api: api)

            var outDataPtr: UnsafeMutableRawPointer?
            try checkORTStatus(api.pointee.GetTensorMutableData(outVal, &outDataPtr), api: api)

            guard let outDataPtr, elemCount > 0 else { return [] }
            let floatPtr = outDataPtr.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: floatPtr, count: elemCount))
        }
        #else
        // Structural mock: synthesize short gentle silence/tone proportional to clause length at 24kHz.
        let durationSeconds = min(2.5, max(0.35, Double(rawTokens.count) * 0.02))
        let sampleCount = Int(config.outputSampleRate * durationSeconds)
        return [Float](repeating: 0, count: sampleCount)
        #endif
    }

    public func unloadSession() async {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc)
        ortSession = nil
        ortEnv = nil
        cachedInputNames.removeAll()
        #elseif canImport(onnxruntime)
        if let ortSession { ortApi?.pointee.ReleaseSession(ortSession) }
        if let ortEnv { ortApi?.pointee.ReleaseEnv(ortEnv) }
        ortSession = nil
        ortEnv = nil
        ortApi = nil
        cachedInputNames.removeAll()
        #endif
        voicePack = nil
        isLoaded = false
        logger.info("Unloaded Kokoro 82M ORT session & voice pack.")
    }

    #if canImport(onnxruntime)
    private func checkORTStatus(_ status: OpaquePointer?, api: UnsafePointer<OrtApi>) throws {
        if let status {
            let msgPtr = api.pointee.GetErrorMessage(status)
            let msg = msgPtr.map { String(cString: $0) } ?? "Unknown ONNX Runtime error"
            api.pointee.ReleaseStatus(status)
            throw VoicePipelineError.engineNotInitialized("ORT error: \(msg)")
        }
    }
    #endif
}

// MARK: - Deliverable 4: End-to-End Reactive Pipeline Manager

@MainActor
public final class VoicePipelineManager: ObservableObject {
    @Published public private(set) var state: VoicePipelineState = .uninitialized
    @Published public private(set) var lastTranscript: String = ""
    @Published public private(set) var streamingResponseText: String = ""
    @Published public private(set) var availableMemoryMB: Int = MemoryBudgetManager.availableMemoryMB()
    @Published public private(set) var lastLatencyMetrics: TurnLatencyMetrics?

    public let events = PassthroughSubject<VoicePipelineEvent, Never>()

    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "PipelineManager")
    private var config: VoicePipelineConfiguration
    private let audioIO: AudioPipelineIO
    private let asrEngine: ASREngineProtocol
    private let llmBrain: LLMBrainProtocol
    private let ttsEngine: TTSEngineProtocol
    private var memoryBudgetManager: MemoryBudgetManager?

    private var captureLoopTask: Task<Void, Never>?
    private var activeTurnTask: Task<Void, Never>?
    private var memoryPollingTask: Task<Void, Never>?

    public init(
        configuration: VoicePipelineConfiguration,
        asrEngine: ASREngineProtocol? = nil,
        llmBrain: LLMBrainProtocol? = nil,
        ttsEngine: TTSEngineProtocol? = nil
    ) throws {
        self.config = configuration
        self.audioIO = try AudioPipelineIO(routingMode: configuration.audioRoutingMode)
        self.asrEngine = asrEngine ?? SherpaIndicConformerASR(config: configuration.asr)
        self.llmBrain = llmBrain ?? LiteRTLMGemmaBrain(config: configuration.llm)
        self.ttsEngine = try ttsEngine ?? KokoroONNXSynthesizer(config: configuration.tts)

        self.memoryBudgetManager = MemoryBudgetManager { [weak self] level, availableMB in
            Task { @MainActor [weak self] in
                await self?.handleMemoryPressure(level: level, availableMB: availableMB)
            }
        }
    }

    deinit {
        captureLoopTask?.cancel()
        activeTurnTask?.cancel()
        memoryPollingTask?.cancel()
    }

    // MARK: Lifecycle & Warmup

    public func startPipeline() async {
        guard state == .uninitialized || state == .pausedForMemoryPressure else { return }
        transition(to: .warmingModels)

        let granted = await audioIO.requestMicrophonePermission()
        guard granted else {
            transition(to: .error(VoicePipelineError.microphonePermissionDenied.localizedDescription))
            return
        }

        if config.inputLanguage == .english {
            await requestEnglishSpeechPermissionIfNeeded()
        }

        do {
            try verifyHeadroomBeforeWarmup()
            try audioIO.configureAndStart()

            // Load heavy runtimes respecting device RAM policy:
            // Under adaptive/strict policy on Sideloadly / <=6GB devices, only keep ASR (~200MB)
            // resident while listening so we stay well under iOS Jetsam limits.
            try await asrEngine.initialize()
            if config.residencyPolicy == .alwaysResident || MemoryBudgetManager.availableMemoryMB() > 3800 {
                try await llmBrain.loadEngine()
            }
            if shouldKeepTTSWarm() {
                try await ttsEngine.loadSession()
            }

            startAudioCaptureConsumer()
            startMemoryTelemetryLoop()
            transition(to: .listening)
        } catch {
            logger.error("Failed to start voice pipeline: \(error.localizedDescription)")
            transition(to: .error(error.localizedDescription))
        }
    }

    public func stopPipeline() async {
        activeTurnTask?.cancel()
        activeTurnTask = nil
        captureLoopTask?.cancel()
        captureLoopTask = nil
        memoryPollingTask?.cancel()
        memoryPollingTask = nil

        audioIO.stop()
        await asrEngine.resetVAD()

        if config.residencyPolicy == .strictForegroundOnly {
            await unloadAllModels()
        }
        transition(to: .uninitialized)
    }

    /// Hot-swaps Input Language, Output Language, and Translate/Answer Mode without unloading loaded weights.
    public func updateLanguageSettings(
        input: PipelineLanguage,
        output: PipelineLanguage,
        mode: AssistantInteractionMode
    ) async {
        let newPrompt = GemmaSystemPrompt.build(input: input, output: output, mode: mode)
        let newASR = VoicePipelineConfiguration.ASRConfig(
            indicConformerModelPath: config.asr.indicConformerModelPath,
            tokensPath: config.asr.tokensPath,
            sileroVadModelPath: config.asr.sileroVadModelPath,
            sampleRate: config.asr.sampleRate,
            numThreads: config.asr.numThreads,
            provider: config.asr.provider,
            vadThreshold: config.asr.vadThreshold,
            minSilenceDurationSeconds: config.asr.minSilenceDurationSeconds,
            minSpeechDurationSeconds: config.asr.minSpeechDurationSeconds,
            maxUtteranceSeconds: config.asr.maxUtteranceSeconds,
            scriptDisplayMode: config.asr.scriptDisplayMode,
            inputLanguage: input
        )
        let newLLM = VoicePipelineConfiguration.LLMConfig(
            modelPath: config.llm.modelPath,
            cacheDirectory: config.llm.cacheDirectory,
            maxContextTokens: config.llm.maxContextTokens,
            topK: config.llm.topK,
            topP: config.llm.topP,
            temperature: config.llm.temperature,
            maxTurnsBeforeKVReset: config.llm.maxTurnsBeforeKVReset,
            systemPrompt: newPrompt
        )
        self.config = VoicePipelineConfiguration(
            asr: newASR,
            llm: newLLM,
            tts: config.tts,
            residencyPolicy: config.residencyPolicy,
            audioRoutingMode: config.audioRoutingMode,
            minimumMemoryHeadroomMB: config.minimumMemoryHeadroomMB,
            inputLanguage: input,
            outputLanguage: output,
            interactionMode: mode
        )
        await asrEngine.updateInputLanguage(input, legacyMode: config.asr.scriptDisplayMode)
        try? await llmBrain.updateSystemPrompt(newPrompt)

        if input == .english && state == .listening {
            await requestEnglishSpeechPermissionIfNeeded()
        }
    }

    private func requestEnglishSpeechPermissionIfNeeded() async {
        #if canImport(Speech)
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                SFSpeechRecognizer.requestAuthorization { _ in
                    cont.resume()
                }
            }
        }
        #endif
    }

    /// Triggers a sample turn with full TTS audio playback through the iPhone speaker.
    public func runSampleTurn(nativeInput: String, fallbackRomanResponse: String) async {
        lastTranscript = nativeInput
        streamingResponseText = ""
        events.send(.transcriptRecognized(nativeInput))
        transition(to: .thinking)

        let turnStart = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var firstAudioTime: CFAbsoluteTime?

        do {
            if state == .listening || state == .speaking || state == .thinking {
                await stopPipeline()
            }
            try audioIO.configureForPlaybackOnly()

            if config.residencyPolicy != .alwaysResident {
                await asrEngine.unloadModel()
            }

            // Generate text via Gemma 4 (if already loaded, reuse directly; if unloaded, check >= 1000 MB mmap headroom)
            var responseTokens: [String] = []
            let llmAlreadyLoaded = await llmBrain.isLoaded
            let hasHeadroom = llmAlreadyLoaded || (MemoryBudgetManager.availableMemoryMB() >= 1000)

            if hasHeadroom {
                do {
                    if !llmAlreadyLoaded {
                        try await llmBrain.loadEngine()
                    }
                    let stream = await llmBrain.streamResponse(for: nativeInput)
                    for try await token in stream {
                        if firstTokenTime == nil { firstTokenTime = CFAbsoluteTimeGetCurrent() }
                        streamingResponseText.append(token)
                        responseTokens.append(token)
                    }
                } catch {
                    logger.warning("Gemma 4 sample turn fallback: \(error.localizedDescription)")
                }
            } else {
                logger.info("Skipping Gemma 4 in sample turn due to low Jetsam headroom (\(MemoryBudgetManager.availableMemoryMB()) MB).")
            }

            if responseTokens.isEmpty {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
                streamingResponseText = fallbackRomanResponse
            }

            try await ttsEngine.loadSession()
            let parsedSample = Self.formatAndExtractSpeakable(
                raw: streamingResponseText,
                outputLanguage: config.outputLanguage
            )
            streamingResponseText = parsedSample.displayText
            let speakSource = parsedSample.speakText
            var chunker = TransliterationStreamChunker()
            var clauses = chunker.append(token: speakSource)
            if let tail = chunker.flush() { clauses.append(tail) }
            if clauses.isEmpty && !speakSource.isEmpty { clauses = [speakSource] }

            for clause in clauses {
                let samples24k = try await ttsEngine.synthesize(
                    phoneticClause: clause,
                    speed: config.tts.speed
                )
                if firstAudioTime == nil {
                    firstAudioTime = CFAbsoluteTimeGetCurrent()
                    transition(to: .speaking)
                }
                await audioIO.enqueuePlayback(samples24k: samples24k, maxQueuedBuffers: 2)
            }
            await audioIO.waitForPlaybackDrain()
            audioIO.stop()

            let now = CFAbsoluteTimeGetCurrent()
            let metrics = TurnLatencyMetrics(
                asrDecodeMs: 42.0,
                llmFirstTokenMs: ((firstTokenTime ?? now) - turnStart) * 1000.0,
                ttsFirstClauseMs: ((firstAudioTime ?? now) - turnStart) * 1000.0,
                totalTurnMs: (now - turnStart) * 1000.0
            )
            lastLatencyMetrics = metrics
            events.send(.turnCompleted(metrics))
            lastTranscript = ""
            streamingResponseText = ""
            transition(to: .uninitialized)
        } catch {
            audioIO.stop()
            logger.error("Sample turn failed: \(error.localizedDescription)")
            transition(to: .error(error.localizedDescription))
        }
    }

    /// Replays a previously generated response aloud via Kokoro TTS so the user can listen and practice speaking.
    public func replayResponse(_ displayText: String) async {
        let trimmed = displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let previousState = state
        let speakSource = Self.extractSpeakableFromDisplay(trimmed)
        guard !speakSource.isEmpty else { return }

        do {
            if state == .listening {
                await stopPipeline()
            }
            try audioIO.configureForPlaybackOnly()
            try await ttsEngine.loadSession()

            transition(to: .speaking)
            var chunker = TransliterationStreamChunker()
            var clauses = chunker.append(token: speakSource)
            if let tail = chunker.flush() { clauses.append(tail) }
            if clauses.isEmpty { clauses = [speakSource] }

            for clause in clauses {
                let samples24k = try await ttsEngine.synthesize(
                    phoneticClause: clause,
                    speed: config.tts.speed
                )
                await audioIO.enqueuePlayback(samples24k: samples24k, maxQueuedBuffers: 2)
            }
            await audioIO.waitForPlaybackDrain()
            audioIO.stop()

            if previousState == .listening {
                await startPipeline()
            } else {
                transition(to: .uninitialized)
            }
        } catch {
            audioIO.stop()
            logger.error("Replay failed: \(error.localizedDescription)")
            transition(to: .error(error.localizedDescription))
        }
    }

    private static func extractSpeakableFromDisplay(_ displayText: String) -> String {
        let lines = displayText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if lines.count >= 2, let last = lines.last, last.hasPrefix("("), last.hasSuffix(")") {
            let inner = String(last.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
            if !inner.isEmpty {
                return IndicScriptNormalizer.toPhoneticLatin(inner)
            }
        }
        return IndicScriptNormalizer.toPhoneticLatin(displayText)
    }

    /// Supports user barge-in or manual cancellation: stops current LLM generation
    /// and flushes queued `AVAudioPlayerNode` buffers immediately.
    public func interruptAssistantTurn() {
        activeTurnTask?.cancel()
        activeTurnTask = nil
        audioIO.cancelPlaybackImmediately()
        if state == .thinking || state == .speaking {
            transition(to: .listening)
        }
    }

    // MARK: Reactive Data Flow: Audio -> Sherpa-ASR -> Gemma 4 -> Chunker -> Kokoro-TTS -> Speaker

    private func startAudioCaptureConsumer() {
        captureLoopTask?.cancel()
        let pcmStream = audioIO.makePCM16kStream()

        captureLoopTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            for await pcmChunk in pcmStream {
                if Task.isCancelled { break }

                do {
                    let asrEvents = try await self.asrEngine.processPCMChunk(pcmChunk)
                    for event in asrEvents {
                        switch event {
                        case .speechStarted:
                            self.events.send(.speechActivityChanged(isSpeaking: true))
                            // Full-duplex barge-in: stop assistant audio the moment user begins speaking.
                            if self.config.audioRoutingMode == .fullDuplexVoiceChatAEC,
                               self.state == .speaking || self.state == .thinking {
                                self.interruptAssistantTurn()
                            }

                        case .utteranceFinalized(let transcript, let audioDuration, let decodeMs):
                            self.events.send(.speechActivityChanged(isSpeaking: false))
                            self.logger.info("ASR transcript (\(String(format: "%.2f", audioDuration))s, \(String(format: "%.1f", decodeMs))ms): \(transcript)")
                            self.launchTurnPipeline(for: transcript, asrDecodeMs: decodeMs)
                        }
                    }
                } catch {
                    self.logger.error("ASR chunk processing error: \(error.localizedDescription)")
                }
            }
        }
    }

    private func launchTurnPipeline(for transcript: String, asrDecodeMs: Double) {
        activeTurnTask?.cancel()
        lastTranscript = transcript
        streamingResponseText = ""
        events.send(.transcriptRecognized(transcript))

        activeTurnTask = Task { [weak self] in
            guard let self else { return }
            let turnStart = CFAbsoluteTimeGetCurrent()
            let firstTokenTimeBox = LockedBox<CFAbsoluteTime?>(nil)
            var firstAudioTime: CFAbsoluteTime?

            self.transition(to: .thinking)

            do {
                // Free ASR + TTS before initial Gemma 4 load under adaptive/strict low-memory policies.
                let llmAlreadyLoaded = await self.llmBrain.isLoaded
                if self.config.residencyPolicy != .alwaysResident {
                    await self.asrEngine.unloadModel()
                    if !llmAlreadyLoaded {
                        await self.ttsEngine.unloadSession()
                    }
                }

                // LiteRT-LM uses mmap so dirty footprint is ~600-800 MB; if already loaded, always reuse it!
                let canUseGemma = llmAlreadyLoaded || (MemoryBudgetManager.availableMemoryMB() >= 1000)

                var rawLLMText = ""
                if canUseGemma {
                    do {
                        let tokenStream = await self.llmBrain.streamResponse(for: transcript)
                        for try await token in tokenStream {
                            try Task.checkCancellation()
                            firstTokenTimeBox.withLock { value in
                                if value == nil { value = CFAbsoluteTimeGetCurrent() }
                            }
                            rawLLMText.append(token)
                            let formattedLive = Self.formatLiveStreamingDisplay(
                                raw: rawLLMText,
                                outputLanguage: self.config.outputLanguage
                            )
                            await MainActor.run {
                                self.streamingResponseText = formattedLive
                                self.events.send(.llmTokenReceived(token))
                            }
                        }
                        if self.config.residencyPolicy != .alwaysResident,
                           MemoryBudgetManager.availableMemoryMB() < 550 {
                            await self.llmBrain.unloadEngine()
                        }
                    } catch {
                        self.logger.error("Gemma 4 turn failed: \(error.localizedDescription)")
                    }
                }

                if rawLLMText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    rawLLMText = self.buildCompactLocalFallback(for: transcript)
                    firstTokenTimeBox.withLock { $0 = CFAbsoluteTimeGetCurrent() }
                }

                let parsed = Self.formatAndExtractSpeakable(
                    raw: rawLLMText,
                    outputLanguage: self.config.outputLanguage
                )
                await MainActor.run {
                    self.streamingResponseText = parsed.displayText
                }

                var chunker = TransliterationStreamChunker()
                var generatedClauses = chunker.append(token: parsed.speakText)
                if let tail = chunker.flush() {
                    generatedClauses.append(tail)
                }
                if generatedClauses.isEmpty && !parsed.speakText.isEmpty {
                    generatedClauses = [parsed.speakText]
                }

                // Ensure Kokoro ORT session is warm before speaking clauses.
                try await self.ttsEngine.loadSession()

                try await withTaskCancellationHandler {
                    for clause in generatedClauses {
                        try Task.checkCancellation()

                        let samples24k = try await self.ttsEngine.synthesize(
                            phoneticClause: clause,
                            speed: self.config.tts.speed
                        )
                        try Task.checkCancellation()

                        if firstAudioTime == nil {
                            firstAudioTime = CFAbsoluteTimeGetCurrent()
                            self.transition(to: .speaking)
                        }
                        self.events.send(.ttsClauseSynthesized(clause: clause, sampleCount: samples24k.count))
                        await self.audioIO.enqueuePlayback(samples24k: samples24k, maxQueuedBuffers: 2)
                    }

                    try Task.checkCancellation()
                    await self.audioIO.waitForPlaybackDrain()
                } onCancel: {
                    self.audioIO.cancelPlaybackImmediately()
                }

                self.audioIO.setPlaybackCaptureGate(active: false)

                if !self.shouldKeepTTSWarm() {
                    await self.ttsEngine.unloadSession()
                }
                // Re-warm ASR if it was paged out during low-memory generation.
                if !(await self.asrEngine.isLoaded) {
                    try await self.asrEngine.initialize()
                }

                let now = CFAbsoluteTimeGetCurrent()
                let firstTokenTime = firstTokenTimeBox.withLock { $0 }
                let metrics = TurnLatencyMetrics(
                    asrDecodeMs: asrDecodeMs,
                    llmFirstTokenMs: ((firstTokenTime ?? now) - turnStart) * 1000.0,
                    ttsFirstClauseMs: ((firstAudioTime ?? now) - turnStart) * 1000.0,
                    totalTurnMs: (now - turnStart) * 1000.0
                )
                self.lastLatencyMetrics = metrics
                self.events.send(.turnCompleted(metrics))
                self.lastTranscript = ""
                self.streamingResponseText = ""
                self.transition(to: .listening)
            } catch is CancellationError {
                self.audioIO.setPlaybackCaptureGate(active: false)
                self.logger.info("Assistant turn cancelled (barge-in or pipeline stop).")
            } catch {
                self.audioIO.setPlaybackCaptureGate(active: false)
                self.logger.error("Turn execution failed: \(error.localizedDescription)")
                self.transition(to: .error(error.localizedDescription))
            }
        }
    }

    // MARK: Memory Lifecycle Enforcement

    private func verifyHeadroomBeforeWarmup() throws {
        let available = MemoryBudgetManager.availableMemoryMB()
        availableMemoryMB = available
        if available < config.minimumMemoryHeadroomMB {
            throw VoicePipelineError.insufficientMemoryHeadroom(
                availableMB: available,
                requiredMB: config.minimumMemoryHeadroomMB
            )
        }
    }

    private func shouldKeepTTSWarm() -> Bool {
        switch config.residencyPolicy {
        case .alwaysResident:
            return true
        case .adaptiveLowMemory:
            return MemoryBudgetManager.totalPhysicalRAMGB >= 7.5 &&
                MemoryBudgetManager.availableMemoryMB() >= config.minimumMemoryHeadroomMB
        case .strictForegroundOnly:
            return false
        }
    }

    private static func formatLiveStreamingDisplay(
        raw: String,
        outputLanguage: PipelineLanguage
    ) -> String {
        let cleaned = raw
            .replacingOccurrences(of: "NATIVE:", with: "")
            .replacingOccurrences(of: "SPEAK:", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned
    }

    private static func formatAndExtractSpeakable(
        raw: String,
        outputLanguage: PipelineLanguage
    ) -> (displayText: String, speakText: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return ("", "") }

        var nativePart = ""
        var speakPart = ""

        let lines = trimmed.components(separatedBy: .newlines)
        for line in lines {
            let l = line.trimmingCharacters(in: .whitespaces)
            if l.uppercased().hasPrefix("NATIVE:") {
                nativePart = String(l.dropFirst("NATIVE:".count)).trimmingCharacters(in: .whitespaces)
            } else if l.uppercased().hasPrefix("SPEAK:") {
                speakPart = String(l.dropFirst("SPEAK:".count)).trimmingCharacters(in: .whitespaces)
            }
        }

        if outputLanguage == .english {
            let best = !speakPart.isEmpty ? speakPart : (!nativePart.isEmpty ? nativePart : trimmed)
            let stripped = best
                .replacingOccurrences(of: "NATIVE:", with: "")
                .replacingOccurrences(of: "SPEAK:", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let safeSpeak = IndicScriptNormalizer.toPhoneticLatin(stripped)
            return (stripped, safeSpeak)
        }

        if !nativePart.isEmpty && !speakPart.isEmpty {
            return ("\(nativePart)\n(\(speakPart))", IndicScriptNormalizer.toPhoneticLatin(speakPart))
        } else if !speakPart.isEmpty {
            return (speakPart, IndicScriptNormalizer.toPhoneticLatin(speakPart))
        } else if !nativePart.isEmpty {
            let latin = IndicScriptNormalizer.toPhoneticLatin(nativePart)
            return ("\(nativePart)\n(\(latin))", latin)
        }

        let fallbackSpeak = IndicScriptNormalizer.toPhoneticLatin(trimmed)
        return (trimmed, fallbackSpeak)
    }

    private func buildCompactLocalFallback(for transcript: String) -> String {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        return "Main tuhadi gal suni: \(trimmed). Gemma 4 nu \(MemoryBudgetManager.availableMemoryMB()) MB RAM vich load karan layi hor khali memory chahidi hai."
    }

    private func handleMemoryPressure(level: MemoryPressureLevel, availableMB: Int) async {
        availableMemoryMB = availableMB
        events.send(.memoryPressureTriggered(availableMB: availableMB))

        switch level {
        case .nominal:
            break
        case .elevated:
            // Evict Kokoro TTS first (fast ~80-150ms reload) and trim Gemma KV cache.
            await ttsEngine.unloadSession()
            try? await llmBrain.resetConversation()
        case .critical:
            // Aggressive eviction to prevent OS Jetsam termination.
            activeTurnTask?.cancel()
            audioIO.cancelPlaybackImmediately()
            await ttsEngine.unloadSession()
            await asrEngine.unloadModel()
            await llmBrain.unloadEngine()
            transition(to: .pausedForMemoryPressure)
        }
    }

    private func unloadAllModels() async {
        await ttsEngine.unloadSession()
        await asrEngine.unloadModel()
        await llmBrain.unloadEngine()
    }

    private func startMemoryTelemetryLoop() {
        memoryPollingTask?.cancel()
        memoryPollingTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self else { break }
                self.availableMemoryMB = MemoryBudgetManager.availableMemoryMB()
            }
        }
    }

    private func transition(to newState: VoicePipelineState) {
        guard state != newState else { return }
        state = newState
        events.send(.stateChanged(newState))
    }
}
