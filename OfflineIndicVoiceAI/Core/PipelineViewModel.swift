//
//  PipelineViewModel.swift
//  OfflineIndicVoiceAI
//
//  Connects ModelAssetStore and VoicePipelineManager to the SwiftUI layer,
//  maintaining conversation history, live streaming telemetry, and dry-run simulation.
//

import Combine
import Foundation
import UIKit

public struct ConversationTurn: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let timestamp: Date
    public let userNativeTranscript: String
    public let assistantRomanisedText: String
    public let metrics: TurnLatencyMetrics?

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        userNativeTranscript: String,
        assistantRomanisedText: String,
        metrics: TurnLatencyMetrics? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.userNativeTranscript = userNativeTranscript
        self.assistantRomanisedText = assistantRomanisedText
        self.metrics = metrics
    }
}

@MainActor
public final class PipelineViewModel: ObservableObject {
    @Published public private(set) var pipelineState: VoicePipelineState = .uninitialized
    @Published public private(set) var isUserSpeaking: Bool = false
    @Published public private(set) var liveUserTranscript: String = ""
    @Published public private(set) var liveAssistantText: String = ""
    @Published public private(set) var history: [ConversationTurn] = []
    @Published public private(set) var availableMemoryMB: Int = MemoryBudgetManager.availableMemoryMB()
    @Published public private(set) var lastMetrics: TurnLatencyMetrics?
    @Published public private(set) var replayingTurnID: UUID?
    @Published public var bannerErrorMessage: String?

    // MARK: - Teacher Interactive Mode State
    @Published public var teacherTopic: TeacherTopic = .greetings
    @Published public private(set) var teacherIsBusy: Bool = false
    @Published public private(set) var teacherIsSpeaking: Bool = false
    @Published public private(set) var teacherIsListening: Bool = false
    @Published public private(set) var teacherListeningLabel: String = ""
    @Published public private(set) var teacherCurrentLesson: TeacherTurnResult?
    @Published public private(set) var teacherLastEvaluation: TeacherPracticeEvaluation?
    @Published public private(set) var teacherLastLearnerInput: String = ""
    @Published public private(set) var teacherPracticeCount: Int = 0
    @Published public var teacherErrorMessage: String?

    private var teacherCaptureTask: Task<Void, Never>?
    private var teacherSpeechTask: Task<Void, Never>?
    private var teacherChallengeIndex: Int = 0
    private var teacherActivePairKey: String = ""

    public let assetStore: ModelAssetStore
    private var manager: VoicePipelineManager?
    private var cancellables = Set<AnyCancellable>()

    /// True if all native frameworks (`SherpaOnnx`, `LiteRTLM`, `onnxruntime`) are linked
    /// AND required weight files are present on disk.
    public var isRunningLiveWeights: Bool {
        #if (canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED) && canImport(LiteRTLM) && (canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc) || canImport(onnxruntime))
        return assetStore.allRequiredAssetsInstalled
        #else
        return false
        #endif
    }

    public init(assetStore: ModelAssetStore) {
        self.assetStore = assetStore
        rebuildPipelineManager()
    }

    public func rebuildPipelineManager() {
        cancellables.removeAll()
        let currentManager = manager
        Task {
            await currentManager?.stopPipeline()
        }

        do {
            let config = assetStore.buildPipelineConfiguration()
            let newManager = try VoicePipelineManager(configuration: config)
            self.manager = newManager
            bind(to: newManager)
            bannerErrorMessage = nil
        } catch {
            bannerErrorMessage = error.localizedDescription
        }
    }

    public func applyLanguageAndModeChanges() {
        Haptics.tap()
        guard let manager else { return }
        let input = assetStore.inputLanguage
        let output = assetStore.outputLanguage
        let mode = assetStore.interactionMode
        Task {
            await manager.updateLanguageSettings(input: input, output: output, mode: mode)
        }
    }

    public func swapLanguages() {
        let currentIn = assetStore.inputLanguage
        let currentOut = assetStore.outputLanguage
        let nextIn = currentOut
        let nextOut = (currentIn == .auto) ? .english : currentIn
        assetStore.inputLanguageRaw = nextIn.rawValue
        assetStore.outputLanguageRaw = nextOut.rawValue
        applyLanguageAndModeChanges()
    }

    public func toggleListening() {
        Haptics.tap()
        guard let manager else {
            rebuildPipelineManager()
            return
        }

        Task {
            switch pipelineState {
            case .uninitialized, .pausedForMemoryPressure, .error:
                await manager.startPipeline()
            case .listening, .transcribing, .thinking, .speaking, .warmingModels:
                await manager.stopPipeline()
            }
        }
    }

    public func interruptSpeaking() {
        Haptics.impact()
        manager?.interruptAssistantTurn()
    }

    public func clearConversation() {
        Haptics.tap()
        history.removeAll()
        liveUserTranscript = ""
        liveAssistantText = ""
        lastMetrics = nil
    }

    public func deleteTurn(_ turn: ConversationTurn) {
        Haptics.tap()
        history.removeAll { $0.id == turn.id }
    }

    public func submitTypedTurn(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        Haptics.tap()
        guard let manager else {
            rebuildPipelineManager()
            return
        }
        let inputLang = assetStore.inputLanguage
        let outputLang = assetStore.outputLanguage
        let mode = assetStore.interactionMode
        let fallbackResp = Self.fallbackReply(for: outputLang, mode: mode)

        Task {
            await manager.updateLanguageSettings(input: inputLang, output: outputLang, mode: mode)
            await manager.runSampleTurn(
                nativeInput: text,
                fallbackRomanResponse: fallbackResp
            )
        }
    }

    /// Triggers a sample utterance matching ANY selected (Input -> Output) language pair and mode, speaking it aloud via Kokoro TTS.
    public func runMockSampleTurn() {
        Haptics.tap()
        guard let manager else {
            rebuildPipelineManager()
            return
        }
        let inputLang = assetStore.inputLanguage
        let outputLang = assetStore.outputLanguage
        let mode = assetStore.interactionMode

        let sampleInput = Self.samplePhrase(for: inputLang)
        let fallbackResp = Self.fallbackReply(for: outputLang, mode: mode)

        Task {
            await manager.updateLanguageSettings(input: inputLang, output: outputLang, mode: mode)
            await manager.runSampleTurn(
                nativeInput: sampleInput,
                fallbackRomanResponse: fallbackResp
            )
        }
    }

    public func replayTurn(_ turn: ConversationTurn) {
        Haptics.tap()
        guard let manager else { return }
        replayingTurnID = turn.id
        Task {
            await manager.replayResponse(turn.assistantRomanisedText)
            replayingTurnID = nil
        }
    }

    // MARK: - Interactive Teacher Session Actions

    public func openTeacherSession() {
        Haptics.tap()
        teacherLastEvaluation = nil
        teacherLastLearnerInput = ""
        teacherErrorMessage = nil
        let pairKey = "\(assetStore.inputLanguage.rawValue)->\(assetStore.outputLanguage.rawValue)"
        if teacherCurrentLesson == nil || teacherActivePairKey != pairKey {
            teacherActivePairKey = pairKey
            teacherChallengeIndex = 0
            loadInstantLessonAndSpeak(includeBilingualGuidance: true)
        }
    }

    public func selectTeacherTopic(_ topic: TeacherTopic) {
        Haptics.tap()
        teacherTopic = topic
        teacherChallengeIndex = 0
        teacherLastEvaluation = nil
        teacherLastLearnerInput = ""
        loadInstantLessonAndSpeak(includeBilingualGuidance: true)
    }

    public func nextTeacherChallenge() {
        Haptics.tap()
        teacherChallengeIndex += 1
        teacherLastEvaluation = nil
        teacherLastLearnerInput = ""
        loadInstantLessonAndSpeak(includeBilingualGuidance: false)
    }

    public func sendTeacherKnownText(_ rawText: String) {
        let clean = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        Haptics.tap()
        teacherLastLearnerInput = clean
        teacherLastEvaluation = nil
        requestTeacherTranslation(learnerInput: clean)
    }

    public func toggleTeacherMic(practicingTarget: Bool) {
        Haptics.tap()
        cancelTeacherSpeech()
        guard let manager else { return }

        if teacherIsListening {
            teacherCaptureTask?.cancel()
            teacherCaptureTask = nil
            teacherIsListening = false
            teacherListeningLabel = ""
            Task {
                await manager.stopTeacherCapture()
            }
            return
        }

        let listenLang = practicingTarget ? assetStore.outputLanguage : assetStore.inputLanguage
        teacherIsListening = true
        teacherListeningLabel = practicingTarget
            ? "Repeat in \(assetStore.outputLanguage.displayName)..."
            : "Speak in \(assetStore.inputLanguage.displayName)..."
        teacherErrorMessage = nil

        teacherCaptureTask = Task {
            do {
                let transcript = try await manager.captureSingleTeacherUtterance(inputLanguage: listenLang)
                self.teacherIsListening = false
                self.teacherListeningLabel = ""
                let normalized = IndicScriptNormalizer.normalize(
                    transcript,
                    inputLanguage: listenLang,
                    legacyMode: self.assetStore.scriptDisplayMode
                )
                let clean = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !clean.isEmpty else { return }

                if practicingTarget, let expectedSpeak = self.teacherCurrentLesson?.targetPhoneticSpeak {
                    let evaluation = TeacherPhoneticMatcher.score(spoken: clean, expectedPhonetic: expectedSpeak)
                    self.teacherLastEvaluation = evaluation
                    if evaluation.matchPercentage >= 50 {
                        self.teacherPracticeCount += 1
                    }
                } else {
                    self.teacherLastLearnerInput = clean
                    self.teacherLastEvaluation = nil
                    self.requestTeacherTranslation(learnerInput: clean)
                }
            } catch is CancellationError {
                self.teacherIsListening = false
                self.teacherListeningLabel = ""
            } catch {
                self.teacherIsListening = false
                self.teacherListeningLabel = ""
                self.teacherErrorMessage = error.localizedDescription
            }
        }
    }

    public func replayTeacherAudio(slow: Bool) {
        Haptics.tap()
        guard let speakText = teacherCurrentLesson?.targetPhoneticSpeak, !speakText.isEmpty else { return }
        let speed: Float = slow ? 0.78 : Float(assetStore.ttsSpeed)
        playTeacherClauses([speakText], speedOverride: speed)
    }

    public func cancelTeacherSpeech() {
        teacherSpeechTask?.cancel()
        teacherSpeechTask = nil
        teacherIsSpeaking = false
        Task {
            await manager?.cancelTeacherPlayback()
        }
    }

    public func closeTeacherSession() {
        teacherSpeechTask?.cancel()
        teacherSpeechTask = nil
        teacherIsSpeaking = false
        teacherCaptureTask?.cancel()
        teacherCaptureTask = nil
        teacherIsListening = false
        teacherListeningLabel = ""
        Task {
            await manager?.cancelTeacherPlayback()
            await manager?.stopTeacherCapture()
            let input = assetStore.inputLanguage
            let output = assetStore.outputLanguage
            let mode = assetStore.interactionMode
            await manager?.updateLanguageSettings(input: input, output: output, mode: mode)
        }
    }

    private func loadInstantLessonAndSpeak(includeBilingualGuidance: Bool) {
        let nativeLang = assetStore.inputLanguage
        let targetLang = assetStore.outputLanguage
        let lesson = VoicePipelineManager.instantTeacherLesson(
            topic: teacherTopic,
            nativeLanguage: nativeLang,
            targetLanguage: targetLang,
            challengeIndex: teacherChallengeIndex
        )
        teacherCurrentLesson = lesson
        teacherIsBusy = false
        teacherErrorMessage = nil

        var clauses: [String] = []
        if includeBilingualGuidance {
            let guidance = VoicePipelineManager.teacherGuidance(
                topic: teacherTopic,
                nativeLanguage: nativeLang,
                targetLanguage: targetLang
            )
            clauses.append(guidance.speakEnglish)
            if let knownSpeak = guidance.speakKnown, !knownSpeak.isEmpty {
                clauses.append(knownSpeak)
            }
        }
        if !lesson.targetPhoneticSpeak.isEmpty {
            clauses.append(lesson.targetPhoneticSpeak)
        }
        playTeacherClauses(clauses, speedOverride: Float(assetStore.ttsSpeed))
    }

    private func requestTeacherTranslation(learnerInput: String) {
        cancelTeacherSpeech()
        guard let manager else { return }
        teacherIsBusy = true
        teacherErrorMessage = nil
        let topic = teacherTopic
        let nativeLang = assetStore.inputLanguage
        let targetLang = assetStore.outputLanguage
        let idx = teacherChallengeIndex

        Task {
            let lesson = await manager.generateTeacherTurn(
                topic: topic,
                nativeLanguage: nativeLang,
                targetLanguage: targetLang,
                learnerInput: learnerInput,
                challengeIndex: idx
            )
            self.teacherCurrentLesson = lesson
            self.teacherIsBusy = false
            self.teacherPracticeCount += 1
            if !lesson.targetPhoneticSpeak.isEmpty {
                self.playTeacherClauses([lesson.targetPhoneticSpeak], speedOverride: Float(assetStore.ttsSpeed))
            }
        }
    }

    private func playTeacherClauses(_ clauses: [String], speedOverride: Float) {
        teacherSpeechTask?.cancel()
        teacherSpeechTask = nil
        guard let manager, !clauses.isEmpty else {
            teacherIsSpeaking = false
            return
        }
        teacherIsSpeaking = true
        teacherSpeechTask = Task {
            await manager.cancelTeacherPlayback()
            if !Task.isCancelled {
                await manager.speakTeacherSequence(clauses, speedOverride: speedOverride)
            }
            if !Task.isCancelled {
                self.teacherIsSpeaking = false
            }
        }
    }

    private static func samplePhrase(for lang: PipelineLanguage) -> String {
        switch lang {
        case .auto, .punjabi: return "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਮੌਸਮ ਕਿਵੇਂ ਹੈ?"
        case .hindi: return "नमस्ते, आज मौसम कैसा है?"
        case .english: return "Hello, how is the weather today?"
        case .bengali: return "নমস্কার, আজ আবহাওয়া কেমন?"
        case .tamil: return "வணக்கம், இன்று வானிலை எப்படி உள்ளது?"
        case .telugu: return "నమస్కారం, ఈ రోజు వాతావరణం ఎలా ఉంది?"
        case .marathi: return "नमस्कार, आज हवामान कसे आहे?"
        case .gujarati: return "નમસ્તે, આજે હવામાન કેવું છે?"
        case .kannada: return "ನಮಸ್ಕಾರ, ಇಂದು ಹವಾಮಾನ ಹೇಗಿದೆ?"
        }
    }

    private static func fallbackReply(for lang: PipelineLanguage, mode: AssistantInteractionMode) -> String {
        switch (lang, mode) {
        case (.english, .translate):
            return "Hello, how is the weather today?"
        case (.english, .smartAnswer):
            return "Hello! I am doing great. The weather looks lovely today."
        case (.punjabi, _), (.auto, _):
            return "NATIVE: ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਮੌਸਮ ਕਿਵੇਂ ਹੈ?\nSPEAK: Sat sri akaal, ajj mausam kiven hai?"
        case (.hindi, _):
            return "NATIVE: नमस्ते, आज मौसम कैसा है?\nSPEAK: Namaste, aaj mausam kaisa hai?"
        case (.bengali, _):
            return "NATIVE: নমস্কার, আজ আবহাওয়া কেমন?\nSPEAK: Nomoshkar, aaj aabohaowa kemon?"
        case (.tamil, _):
            return "NATIVE: வணக்கம், இன்று வானிலை எப்படி உள்ளது?\nSPEAK: Vanakkam, indru vaanilai eppadi ulladhu?"
        case (.telugu, _):
            return "NATIVE: నమస్కారం, ఈ రోజు వాతావరణం ఎలా ఉంది?\nSPEAK: Namaskaram, ee roju vaataavaranam ela undi?"
        case (.marathi, _):
            return "NATIVE: नमस्कार, आज हवामान कसे आहे?\nSPEAK: Namaskar, aaj hawaamaan kase aahe?"
        case (.gujarati, _):
            return "NATIVE: નમસ્તે, આજે હવામાન કેવું છે?\nSPEAK: Namaste, aaje havaamaan kevu chhe?"
        case (.kannada, _):
            return "NATIVE: ನಮಸ್ಕಾರ, ಇಂದು ಹವಾಮಾನ ಹೇಗಿದೆ?\nSPEAK: Namaskara, indu havaamaana hegide?"
        }
    }

    private func bind(to manager: VoicePipelineManager) {
        manager.$state
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newState in
                self?.pipelineState = newState
                if case .error(let msg) = newState {
                    self?.bannerErrorMessage = msg
                }
            }
            .store(in: &cancellables)

        manager.$availableMemoryMB
            .receive(on: DispatchQueue.main)
            .assign(to: &$availableMemoryMB)

        manager.$lastTranscript
            .receive(on: DispatchQueue.main)
            .assign(to: &$liveUserTranscript)

        manager.$streamingResponseText
            .receive(on: DispatchQueue.main)
            .assign(to: &$liveAssistantText)

        manager.events
            .receive(on: DispatchQueue.main)
            .sink { [weak self] event in
                guard let self else { return }
                switch event {
                case .speechActivityChanged(let speaking):
                    self.isUserSpeaking = speaking
                case .turnCompleted(let metrics):
                    self.lastMetrics = metrics
                    let userText = self.liveUserTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
                    let aiText = self.liveAssistantText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !userText.isEmpty || !aiText.isEmpty {
                        self.history.append(
                            ConversationTurn(
                                userNativeTranscript: userText,
                                assistantRomanisedText: aiText,
                                metrics: metrics
                            )
                        )
                    }
                    self.liveUserTranscript = ""
                    self.liveAssistantText = ""
                case .memoryPressureTriggered(let availMB):
                    self.availableMemoryMB = availMB
                default:
                    break
                }
            }
            .store(in: &cancellables)
    }
}

private enum Haptics {
    static func tap() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    static func impact() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }
}
