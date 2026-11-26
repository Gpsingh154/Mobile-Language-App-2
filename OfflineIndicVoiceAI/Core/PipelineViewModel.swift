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
    @Published public var bannerErrorMessage: String?

    public let assetStore: ModelAssetStore
    private var manager: VoicePipelineManager?
    private var cancellables = Set<AnyCancellable>()

    /// True if all native frameworks (`SherpaOnnx`, `LiteRTLM`, `OnnxRuntimeBindings`) are linked
    /// AND required weight files are present on disk.
    public var isRunningLiveWeights: Bool {
        #if (canImport(SherpaOnnx) || canImport(sherpa_onnx) || SHERPA_ONNX_ENABLED) && canImport(LiteRTLM) && (canImport(OnnxRuntimeBindings) || canImport(onnxruntime_objc))
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

    /// Triggers a simulated 16 kHz Punjabi utterance when evaluating UI/pipeline wiring
    /// in structural mock mode without downloaded model weights.
    public func runMockSampleTurn() {
        Haptics.tap()
        Task {
            let previousState = pipelineState
            let sampleNativeInput = "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਮੌਸਮ ਕਿਵੇਂ ਹੈ?"
            liveUserTranscript = sampleNativeInput
            liveAssistantText = ""
            pipelineState = .thinking

            let brain = LiteRTLMGemmaBrain(config: assetStore.buildPipelineConfiguration().llm)
            var chunker = TransliterationStreamChunker()
            let start = CFAbsoluteTimeGetCurrent()
            var firstTokenMs: Double = 0

            do {
                try await brain.loadEngine()
                for try await token in await brain.streamResponse(for: sampleNativeInput) {
                    if firstTokenMs == 0 {
                        firstTokenMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                    }
                    liveAssistantText.append(token)
                    _ = chunker.append(token: token)
                }
                _ = chunker.flush()
                let totalMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                let metrics = TurnLatencyMetrics(
                    asrDecodeMs: 42.0,
                    llmFirstTokenMs: firstTokenMs,
                    ttsFirstClauseMs: firstTokenMs + 65.0,
                    totalTurnMs: totalMs
                )
                lastMetrics = metrics
                history.append(
                    ConversationTurn(
                        userNativeTranscript: sampleNativeInput,
                        assistantRomanisedText: liveAssistantText,
                        metrics: metrics
                    )
                )
                pipelineState = (previousState == .listening) ? .listening : .uninitialized
            } catch {
                bannerErrorMessage = error.localizedDescription
                pipelineState = .error(error.localizedDescription)
            }
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
