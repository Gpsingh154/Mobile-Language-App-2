//
//  VoiceAssistantView.swift
//  OfflineIndicVoiceAI
//
//  Main iPhone voice-to-voice interface displaying live ASR native script,
//  streaming Gemma 4 Romanised output, Jetsam RAM telemetry, and audio orb controls.
//

import SwiftUI

struct VoiceAssistantView: View {
    @ObservedObject var viewModel: PipelineViewModel
    @ObservedObject var assetStore: ModelAssetStore

    @State private var showModelSetup = false
    @State private var showSettings = false

    private var isActiveSession: Bool {
        switch viewModel.pipelineState {
        case .listening, .transcribing, .thinking, .speaking, .warmingModels:
            return true
        case .uninitialized, .pausedForMemoryPressure, .error:
            return false
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                telemetryHeader

                languageAndModeBar

                if !viewModel.isRunningLiveWeights {
                    setupNoticeCard
                }

                if let errorText = viewModel.bannerErrorMessage {
                    errorBanner(errorText)
                }

                VoiceStateOrb(
                    state: viewModel.pipelineState,
                    isUserSpeaking: viewModel.isUserSpeaking
                )
                .padding(.vertical, 2)

                conversationFeed

                bottomControlDock
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)
            .padding(.bottom, 8)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background {
                LinearGradient(
                    colors: [Color(red: 0.05, green: 0.07, blue: 0.12), Color(red: 0.08, green: 0.10, blue: 0.17)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
            }
            .navigationTitle("Vaani Local")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showModelSetup = true
                    } label: {
                        Label("Models", systemImage: "square.and.arrow.down")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Label("Settings", systemImage: "slider.horizontal.3")
                    }
                }
            }
            .sheet(isPresented: $showModelSetup) {
                ModelSetupView(assetStore: assetStore) {
                    viewModel.rebuildPipelineManager()
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(assetStore: assetStore) {
                    viewModel.rebuildPipelineManager()
                }
            }
        }
    }

    private var telemetryHeader: some View {
        HStack(spacing: 10) {
            StatusPill(
                icon: viewModel.isRunningLiveWeights ? "bolt.shield.fill" : "cpu",
                text: viewModel.isRunningLiveWeights ? "100% Offline Weights" : "Mock / Structural Mode",
                tint: viewModel.isRunningLiveWeights ? .green : .orange
            )

            Spacer()

            StatusPill(
                icon: "memorychip",
                text: "\(viewModel.availableMemoryMB) MB free",
                tint: viewModel.availableMemoryMB < 450 ? .red : .cyan
            )
        }
        .padding(.top, 2)
    }

    private var languageAndModeBar: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Menu {
                    ForEach(PipelineLanguage.allCases) { lang in
                        Button {
                            assetStore.inputLanguageRaw = lang.rawValue
                            viewModel.applyLanguageAndModeChanges()
                        } label: {
                            HStack {
                                Text(lang.shortChipLabel)
                                if assetStore.inputLanguage == lang {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Speak:")
                            .foregroundStyle(.secondary)
                        Text(assetStore.inputLanguage.displayName)
                            .fontWeight(.semibold)
                            .foregroundStyle(.cyan)
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .padding(.horizontal, 8)
                    .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                }

                Button {
                    viewModel.swapLanguages()
                } label: {
                    Image(systemName: "arrow.left.arrow.right")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.white)
                        .padding(8)
                        .background(Color.indigo.opacity(0.45), in: Circle())
                }

                Menu {
                    ForEach(PipelineLanguage.outputCases) { lang in
                        Button {
                            assetStore.outputLanguageRaw = lang.rawValue
                            viewModel.applyLanguageAndModeChanges()
                        } label: {
                            HStack {
                                Text(lang.shortChipLabel)
                                if assetStore.outputLanguage == lang {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Reply:")
                            .foregroundStyle(.secondary)
                        Text(assetStore.outputLanguage.displayName)
                            .fontWeight(.semibold)
                            .foregroundStyle(.mint)
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .padding(.horizontal, 8)
                    .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                }
            }

            HStack(spacing: 6) {
                ForEach(AssistantInteractionMode.allCases) { mode in
                    let isSelected = (assetStore.interactionMode == mode)
                    Button {
                        assetStore.interactionModeRaw = mode.rawValue
                        viewModel.applyLanguageAndModeChanges()
                    } label: {
                        Label(mode.title, systemImage: mode.iconName)
                            .font(.caption.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                            .foregroundStyle(isSelected ? .white : .secondary)
                            .background(
                                isSelected ? Color.indigo.opacity(0.85) : Color.white.opacity(0.05),
                                in: RoundedRectangle(cornerRadius: 9)
                            )
                    }
                }
            }
        }
    }

    private var setupNoticeCard: some View {
        HStack(spacing: 12) {
            Image(systemName: "externaldrive.badge.plus")
                .font(.title3)
                .foregroundStyle(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text(assetStore.allRequiredAssetsInstalled ? "Link Native SDKs in Xcode" : "Offline Model Files Needed")
                    .font(.subheadline.weight(.semibold))
                Text(assetStore.allRequiredAssetsInstalled
                     ? "Models are ready on disk. Build in Xcode with SherpaOnnx, LiteRTLM & ORT linked."
                     : "Import or download IndicConformer, Gemma 4 (.litertlm), and Kokoro 82M to run live.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Setup") {
                showModelSetup = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .lineLimit(2)
            Spacer()
            Button("Dismiss") {
                viewModel.bannerErrorMessage = nil
            }
            .font(.caption.weight(.semibold))
        }
        .padding(10)
        .background(Color.red.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
    }

    private var conversationFeed: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if viewModel.history.isEmpty && viewModel.liveUserTranscript.isEmpty {
                        emptyPromptPlaceholder
                    }

                    ForEach(viewModel.history) { turn in
                        TurnCardView(turn: turn)
                            .id(turn.id)
                    }

                    if !viewModel.liveUserTranscript.isEmpty || !viewModel.liveAssistantText.isEmpty || viewModel.isUserSpeaking {
                        activeStreamingCard
                            .id("active-turn")
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: viewModel.history.count) { _ in
                if let lastID = viewModel.history.last?.id {
                    withAnimation { proxy.scrollTo(lastID, anchor: .bottom) }
                }
            }
            .onChange(of: viewModel.liveAssistantText) { _ in
                withAnimation { proxy.scrollTo("active-turn", anchor: .bottom) }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyPromptPlaceholder: some View {
        VStack(spacing: 8) {
            Text("Speak in Punjabi, Hindi, Tamil, Bengali, or English")
                .font(.subheadline.weight(.medium))
                .multilineTextAlignment(.center)
            Text("IndicConformer transcribes native script -> Gemma 4 responds in phonetic Roman text -> Kokoro 82M speaks naturally.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(20)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 16))
    }

    private var activeStreamingCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.isUserSpeaking && viewModel.liveUserTranscript.isEmpty {
                Label("Listening to your voice (16 kHz)...", systemImage: "waveform")
                    .font(.subheadline)
                    .foregroundStyle(.mint)
            }
            if !viewModel.liveUserTranscript.isEmpty {
                Label(viewModel.liveUserTranscript, systemImage: "waveform")
                    .font(.subheadline)
                    .foregroundStyle(.cyan)
            }
            if !viewModel.liveAssistantText.isEmpty {
                Text(viewModel.liveAssistantText)
                    .font(.body)
                    .foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.indigo.opacity(0.22), in: RoundedRectangle(cornerRadius: 14))
    }

    private var bottomControlDock: some View {
        VStack(spacing: 10) {
            Button {
                viewModel.runMockSampleTurn()
            } label: {
                Label("Test Sample Punjabi Turn", systemImage: "play.circle.fill")
                    .font(.subheadline.weight(.medium))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .tint(.cyan)
            .disabled(isActiveSession)

            HStack(spacing: 18) {
                Button {
                    viewModel.clearConversation()
                } label: {
                    Image(systemName: "trash")
                        .font(.title3)
                        .frame(width: 48, height: 48)
                }
                .buttonStyle(.bordered)
                .clipShape(Circle())
                .disabled(viewModel.history.isEmpty && viewModel.liveUserTranscript.isEmpty)

                Button {
                    viewModel.toggleListening()
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: isActiveSession ? "stop.fill" : "mic.fill")
                            .font(.title2)
                        Text(isActiveSession ? "Stop" : "Start Voice")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                }
                .buttonStyle(.borderedProminent)
                .tint(isActiveSession ? .red : .indigo)
                .clipShape(Capsule())

                if viewModel.pipelineState == .thinking || viewModel.pipelineState == .speaking {
                    Button {
                        viewModel.interruptSpeaking()
                    } label: {
                        Image(systemName: "hand.raised.fill")
                            .font(.title3)
                            .frame(width: 48, height: 48)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .clipShape(Circle())
                }
            }
        }
    }
}

private struct StatusPill: View {
    let icon: String
    let text: String
    let tint: Color

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .foregroundStyle(tint)
            Text(text)
                .lineLimit(1)
                .font(.caption2.weight(.semibold))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Color.white.opacity(0.08), in: Capsule())
    }
}

private struct VoiceStateOrb: View {
    let state: VoicePipelineState
    let isUserSpeaking: Bool

    @State private var pulse = false

    private var orbColor: Color {
        switch state {
        case .uninitialized: return .gray
        case .warmingModels: return .orange
        case .listening: return isUserSpeaking ? .mint : .cyan
        case .transcribing: return .teal
        case .thinking: return .amberColor
        case .speaking: return .indigo
        case .pausedForMemoryPressure, .error: return .red
        }
    }

    private var statusTitle: String {
        switch state {
        case .uninitialized: return "Tap Start Voice to Listen"
        case .warmingModels: return "Loading Local Models..."
        case .listening: return isUserSpeaking ? "Hearing Speech (16kHz PCM)..." : "Listening Locally..."
        case .transcribing: return "Transcribing Indic Script..."
        case .thinking: return "Gemma 4 Transliterating..."
        case .speaking: return "Kokoro Speaking (24kHz)..."
        case .pausedForMemoryPressure: return "Paused (Memory Guard)"
        case .error: return "Pipeline Error"
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(orbColor.opacity(0.22))
                    .frame(width: 92, height: 92)
                    .scaleEffect(pulse ? 1.15 : 0.94)
                    .animation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true), value: pulse)

                Circle()
                    .fill(
                        RadialGradient(
                            colors: [orbColor.opacity(0.9), orbColor.opacity(0.35)],
                            center: .center,
                            startRadius: 5,
                            endRadius: 38
                        )
                    )
                    .frame(width: 68, height: 68)
                    .shadow(color: orbColor.opacity(0.6), radius: 12)

                Image(systemName: iconName)
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .onAppear { pulse = true }

            Text(statusTitle)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white.opacity(0.9))
        }
    }

    private var iconName: String {
        switch state {
        case .uninitialized: return "mic.slash"
        case .warmingModels: return "hourglass"
        case .listening: return "waveform"
        case .transcribing: return "text.bubble"
        case .thinking: return "brain.head.profile"
        case .speaking: return "speaker.wave.3.fill"
        case .pausedForMemoryPressure: return "memorychip"
        case .error: return "exclamationmark.triangle"
        }
    }
}

private struct TurnCardView: View {
    let turn: ConversationTurn

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !turn.userNativeTranscript.isEmpty {
                Text(turn.userNativeTranscript)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.cyan)
            }
            Text(turn.assistantRomanisedText)
                .font(.body)
                .foregroundStyle(.white)

            if let metrics = turn.metrics {
                HStack(spacing: 10) {
                    Text("ASR \(Int(metrics.asrDecodeMs))ms")
                    Text("TTFT \(Int(metrics.llmFirstTokenMs))ms")
                    Text("TTS \(Int(metrics.ttsFirstClauseMs))ms")
                    Text("Total \(Int(metrics.totalTurnMs))ms")
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
    }
}

private extension Color {
    static let amberColor = Color(red: 1.0, green: 0.72, blue: 0.22)
}
