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
