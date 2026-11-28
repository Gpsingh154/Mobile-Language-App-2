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
    @State private var showTypeInput = false
    @State private var showTeacherSheet = false

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
            .sheet(isPresented: $showTypeInput) {
                TypeInputSheet(
                    inputLanguage: assetStore.inputLanguage,
                    outputLanguage: assetStore.outputLanguage,
                    interactionMode: assetStore.interactionMode
                ) { typedText in
                    viewModel.submitTypedTurn(typedText)
                }
            }
            .sheet(isPresented: $showTeacherSheet, onDismiss: {
                viewModel.closeTeacherSession()
            }) {
                TeacherSessionView(viewModel: viewModel, assetStore: assetStore)
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

            Button {
                viewModel.openTeacherSession()
                showTeacherSheet = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "graduationcap.fill")
                        .foregroundStyle(Color.amberColor)
                    Text("Teacher")
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                    Text("· Practice \(assetStore.outputLanguage.displayName)")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(Color.amberColor)
                }
                .font(.caption)
                .padding(.vertical, 7)
                .padding(.horizontal, 12)
                .background(
                    LinearGradient(
                        colors: [Color.amberColor.opacity(0.20), Color.indigo.opacity(0.22)],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: RoundedRectangle(cornerRadius: 10)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.amberColor.opacity(0.35), lineWidth: 1)
                )
            }
            .disabled(isActiveSession)
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
                        TurnCardView(
                            turn: turn,
                            isReplaying: viewModel.replayingTurnID == turn.id,
                            isBusy: isActiveSession || viewModel.replayingTurnID != nil,
                            onReplay: {
                                viewModel.replayTurn(turn)
                            },
                            onDelete: {
                                viewModel.deleteTurn(turn)
                            }
                        )
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
                Label("Test Sample (\(assetStore.inputLanguage.displayName) → \(assetStore.outputLanguage.displayName))", systemImage: "play.circle.fill")
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .tint(.cyan)
            .disabled(isActiveSession || viewModel.replayingTurnID != nil)

            HStack(spacing: 12) {
                Button {
                    viewModel.clearConversation()
                } label: {
                    Image(systemName: "trash")
                        .font(.title3)
                        .frame(width: 46, height: 46)
                }
                .buttonStyle(.bordered)
                .clipShape(Circle())
                .disabled(viewModel.history.isEmpty && viewModel.liveUserTranscript.isEmpty)

                Button {
                    showTypeInput = true
                } label: {
                    Image(systemName: "keyboard")
                        .font(.title3)
                        .frame(width: 46, height: 46)
                }
                .buttonStyle(.bordered)
                .tint(.cyan)
                .clipShape(Circle())
                .disabled(isActiveSession || viewModel.replayingTurnID != nil)
                .accessibilityLabel("Type text input")

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
    let isReplaying: Bool
    let isBusy: Bool
    let onReplay: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
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

            VStack(spacing: 12) {
                if !turn.assistantRomanisedText.isEmpty {
                    Button(action: onReplay) {
                        Image(systemName: isReplaying ? "speaker.wave.3.fill" : "play.circle.fill")
                            .font(.title2)
                            .foregroundStyle(isReplaying ? .mint : .cyan)
                    }
                    .buttonStyle(.plain)
                    .disabled(isBusy)
                    .accessibilityLabel("Replay pronunciation")
                }

                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                        .font(.subheadline)
                        .foregroundStyle(.red.opacity(0.85))
                        .padding(6)
                        .background(Color.white.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Delete turn")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Language-Mapped Text Input & Built-in Indic Keypad

private struct TypeInputSheet: View {
    let inputLanguage: PipelineLanguage
    let outputLanguage: PipelineLanguage
    let interactionMode: AssistantInteractionMode
    let onSubmit: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text: String = ""
    @State private var useBuiltInIndicKeypad: Bool = false
    @State private var activeTabIndex: Int = 0 // 0: Consonants, 1: Vowels & Matras

    private var hasSystemKeyboardForLanguage: Bool {
        if inputLanguage == .english || inputLanguage == .auto { return true }
        return UITextInputMode.activeInputModes.contains { mode in
            mode.primaryLanguage?.lowercased().hasPrefix(inputLanguage.rawValue) == true
        }
    }

    private var supportsBuiltInKeypad: Bool {
        inputLanguage != .english && inputLanguage != .auto
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                HStack {
                    Label("\(inputLanguage.shortChipLabel) → \(outputLanguage.displayName)", systemImage: "character.bubble")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.cyan)
                    Spacer()
                    Text(interactionMode.shortTitle)
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.indigo.opacity(0.35), in: Capsule())
                }
                .padding(.horizontal)

                LanguageMappedTextField(
                    text: $text,
                    placeholder: "Type in \(inputLanguage.displayName)...",
                    languageCode: inputLanguage.rawValue,
                    disableSystemKeyboard: useBuiltInIndicKeypad,
                    onSubmit: submitAndClose
                )
                .frame(height: 52)
                .padding(.horizontal, 12)
                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)

                if supportsBuiltInKeypad {
                    HStack(spacing: 10) {
                        Button {
                            useBuiltInIndicKeypad.toggle()
                        } label: {
                            Label(
                                useBuiltInIndicKeypad ? "Switch to iOS System Keyboard" : "Open \(inputLanguage.displayName) Keypad",
                                systemImage: "keyboard"
                            )
                            .font(.caption.weight(.medium))
                        }
                        .buttonStyle(.bordered)
                        .tint(.cyan)

                        Spacer()

                        Button {
                            submitAndClose()
                        } label: {
                            Label("Send", systemImage: "paperplane.fill")
                                .font(.subheadline.weight(.semibold))
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.indigo)
                        .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(.horizontal)
                } else {
                    HStack {
                        Spacer()
                        Button {
                            submitAndClose()
                        } label: {
                            Label("Send", systemImage: "paperplane.fill")
                                .font(.subheadline.weight(.semibold))
                                .padding(.horizontal, 12)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.indigo)
                        .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(.horizontal)
                }

                if useBuiltInIndicKeypad && supportsBuiltInKeypad {
                    IndicKeypadView(
                        language: inputLanguage,
                        activeTabIndex: $activeTabIndex,
                        onInsert: { char in text.append(char) },
                        onBackspace: {
                            if !text.isEmpty { text.removeLast() }
                        },
                        onSpace: { text.append(" ") },
                        onSubmit: submitAndClose
                    )
                }

                Spacer()
            }
            .padding(.top, 12)
            .background(Color(red: 0.06, green: 0.08, blue: 0.14).ignoresSafeArea())
            .navigationTitle("Type in \(inputLanguage.displayName)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear {
                // Automatically pop up the built-in language keypad if iOS doesn't have this language keyboard enabled in Settings
                if supportsBuiltInKeypad && !hasSystemKeyboardForLanguage {
                    useBuiltInIndicKeypad = true
                }
            }
        }
        .presentationDetents(useBuiltInIndicKeypad ? [.medium, .large] : [.medium])
    }

    private func submitAndClose() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        dismiss()
        onSubmit(trimmed)
    }
}

private final class PreferredLanguageUITextField: UITextField {
    var preferredLanguageCode: String = "en"

    override var textInputMode: UITextInputMode? {
        let prefix = preferredLanguageCode.lowercased()
        if prefix != "auto" {
            if let matched = UITextInputMode.activeInputModes.first(where: {
                $0.primaryLanguage?.lowercased().hasPrefix(prefix) == true
            }) {
                return matched
            }
        }
        return super.textInputMode
    }
}

private struct LanguageMappedTextField: UIViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let languageCode: String
    let disableSystemKeyboard: Bool
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> PreferredLanguageUITextField {
        let tf = PreferredLanguageUITextField()
        tf.preferredLanguageCode = languageCode
        tf.delegate = context.coordinator
        tf.textColor = .white
        tf.tintColor = .systemCyan
        tf.font = .preferredFont(forTextStyle: .body)
        tf.attributedPlaceholder = NSAttributedString(
            string: placeholder,
            attributes: [.foregroundColor: UIColor.lightGray]
        )
        tf.returnKeyType = .send
        tf.addTarget(context.coordinator, action: #selector(Coordinator.textDidChange(_:)), for: .editingChanged)
        return tf
    }

    func updateUIView(_ uiView: PreferredLanguageUITextField, context: Context) {
        if uiView.text != text {
            uiView.text = text
        }
        let langChanged = (uiView.preferredLanguageCode != languageCode)
        let inputModeChanged = ((uiView.inputView != nil) != disableSystemKeyboard)
        uiView.preferredLanguageCode = languageCode

        if inputModeChanged {
            uiView.inputView = disableSystemKeyboard ? UIView() : nil
        }
        if langChanged || inputModeChanged {
            uiView.reloadInputViews()
        }
        if !disableSystemKeyboard && !uiView.isFirstResponder {
            DispatchQueue.main.async {
                uiView.becomeFirstResponder()
            }
        }
    }

    final class Coordinator: NSObject, UITextFieldDelegate {
        var parent: LanguageMappedTextField
        init(_ parent: LanguageMappedTextField) { self.parent = parent }

        @objc func textDidChange(_ sender: UITextField) {
            parent.text = sender.text ?? ""
        }

        func textFieldShouldReturn(_ textField: UITextField) -> Bool {
            parent.onSubmit()
            return true
        }
    }
}

private struct IndicKeypadView: View {
    let language: PipelineLanguage
    @Binding var activeTabIndex: Int
    let onInsert: (String) -> Void
    let onBackspace: () -> Void
    let onSpace: () -> Void
    let onSubmit: () -> Void

    private var scriptPalette: (consonants: [String], vowelsAndMatras: [String]) {
        switch language {
        case .punjabi:
            return (
                ["ਕ","ਖ","ਗ","ਘ","ਙ","ਚ","ਛ","ਜ","ਝ","ਞ","ਟ","ਠ","ਡ","ਢ","ਣ","ਤ","ਥ","ਦ","ਧ","ਨ","ਪ","ਫ","ਬ","ਭ","ਮ","ਯ","ਰ","ਲ","ਵ","ਸ਼","ਸ","ਹ","ੜ","ਖ਼","ਗ਼","ਜ਼","ਫ਼"],
                ["ਅ","ਆ","ਇ","ਈ","ਉ","ਊ","ਏ","ਐ","ਓ","ਔ","ਾ","ਿ","ੀ","ੁ","ੂ","ੇ","ੈ","ੋ","ੌ","ੰ","ਂ","ੱ","੍","।"]
            )
        case .hindi, .marathi:
            return (
                ["क","ख","ग","घ","च","छ","ज","झ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व","श","ष","स","ह","ळ","ड़","ढ़","ज़","फ़"],
                ["अ","आ","इ","ई","उ","ऊ","ए","ऐ","ओ","औ","ा","ि","ी","ु","ू","े","ै","ो","ौ","ं","ँ","्","ः","।"]
            )
        case .bengali:
            return (
                ["ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ","ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন","প","ফ","ব","ভ","ম","য","র","ল","শ","ষ","স","হ","ড়","ঢ়","য়","ৎ"],
                ["অ","আ","ই","ঈ","উ","ঊ","এ","ঐ","ও","ঔ","া","ি","ী","ু","ূ","ে","ৈ","ো","ৌ","ং","ঁ","্","ঃ","।"]
            )
        case .tamil:
            return (
                ["க","ங","ச","ஞ","ட","ண","த","ந","ப","ம","ய","ர","ல","வ","ழ","ள","ற","ன","ஜ","ஷ","ஸ","ஹ"],
                ["அ","ஆ","இ","ஈ","உ","ஊ","எ","ஏ","ஐ","ஒ","ஓ","ஔ","ா","ி","ீ","ு","ூ","ெ","ே","ை","ொ","ோ","ௌ","்","ஃ"]
            )
        case .telugu:
            return (
                ["క","ఖ","గ","ఘ","చ","ఛ","జ","ఝ","ట","ఠ","డ","ఢ","ణ","త","థ","ద","ధ","న","ప","ఫ","బ","భ","మ","య","ర","ల","వ","శ","ష","స","హ","ళ"],
                ["అ","ఆ","ఇ","ఈ","ఉ","ఊ","ఎ","ఏ","ఐ","ఒ","ఓ","ఔ","ా","ి","ీ","ు","ూ","ె","ే","ై","ొ","ో","ౌ","ం","్","ః"]
            )
        case .gujarati:
            return (
                ["ક","ખ","ગ","ઘ","ચ","છ","જ","ઝ","ટ","ઠ","ડ","ઢ","ણ","ત","થ","દ","ધ","ન","પ","ફ","બ","ભ","મ","ય","ર","લ","વ","શ","ષ","સ","હ","ળ"],
                ["અ","આ","ઇ","ઈ","ઉ","ઊ","એ","ઐ","ઓ","ઔ","ા","િ","ી","ુ","ૂ","ે","ૈ","ો","ૌ","ં","્","ઃ"]
            )
        case .kannada:
            return (
                ["ಕ","ಖ","ಗ","ಘ","ಚ","ಛ","ಜ","ಝ","ಟ","ಠ","ಡ","ಢ","ಣ","ತ","ಥ","ದ","ಧ","ನ","ಪ","ಫ","ಬ","ಭ","ಮ","ಯ","ರ","ಲ","ವ","ಶ","ಷ","ಸ","ಹ","ಳ"],
                ["ಅ","ಆ","ಇ","ಈ","ಉ","ಊ","ಎ","ಏ","ಐ","ಒ","ಓ","ಔ","ಾ","ಿ","ೀ","ು","ೂ","ೆ","ೇ","ೈ","ೊ","ೋ","ೌ","ಂ","್","ಃ"]
            )
        default:
            return ([], [])
        }
    }

    var body: some View {
        VStack(spacing: 8) {
            Picker("Characters", selection: $activeTabIndex) {
                Text("Consonants").tag(0)
                Text("Vowels & Matras").tag(1)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            let keys = (activeTabIndex == 0) ? scriptPalette.consonants : scriptPalette.vowelsAndMatras
            let columns = Array(repeating: GridItem(.flexible(), spacing: 5), count: 8)

            ScrollView {
                LazyVGrid(columns: columns, spacing: 6) {
                    ForEach(keys, id: \.self) { ch in
                        Button {
                            onInsert(ch)
                        } label: {
                            Text(ch)
                                .font(.title3)
                                .foregroundStyle(.white)
                                .frame(maxWidth: .infinity, minHeight: 38)
                                .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                .padding(.horizontal, 10)
            }
            .frame(maxHeight: 220)

            HStack(spacing: 8) {
                Button(action: onSpace) {
                    Text("Space")
                        .font(.subheadline.weight(.medium))
                        .frame(maxWidth: .infinity, minHeight: 40)
                        .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 9))
                        .foregroundStyle(.white)
                }

                Button(action: onBackspace) {
                    Image(systemName: "delete.left.fill")
                        .font(.title3)
                        .frame(width: 64, height: 40)
                        .background(Color.red.opacity(0.28), in: RoundedRectangle(cornerRadius: 9))
                        .foregroundStyle(.white)
                }

                Button(action: onSubmit) {
                    Image(systemName: "paperplane.fill")
                        .font(.title3)
                        .frame(width: 64, height: 40)
                        .background(Color.indigo, in: RoundedRectangle(cornerRadius: 9))
                        .foregroundStyle(.white)
                }
            }
            .padding(.horizontal, 10)
        }
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 8)
    }
}

private extension Color {
    static let amberColor = Color(red: 1.0, green: 0.72, blue: 0.22)
}

// MARK: - Interactive Teacher Mode View

private struct TeacherSessionView: View {
    @ObservedObject var viewModel: PipelineViewModel
    @ObservedObject var assetStore: ModelAssetStore

    @Environment(\.dismiss) private var dismiss
    @State private var showKnownKeyboardSheet = false

    private var knownLanguage: PipelineLanguage {
        assetStore.inputLanguage == .auto ? .punjabi : assetStore.inputLanguage
    }

    private var targetLanguage: PipelineLanguage {
        assetStore.outputLanguage
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                // Header status: language pair & streak counter
                HStack {
                    Label("\(knownLanguage.displayName) → Learn \(targetLanguage.displayName)", systemImage: "graduationcap.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Color.amberColor)
                    Spacer()
                    HStack(spacing: 4) {
                        Image(systemName: "flame.fill")
                            .foregroundStyle(.orange)
                        Text("\(viewModel.teacherPracticeCount) practiced")
                            .font(.caption2.weight(.semibold).monospacedDigit())
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.08), in: Capsule())
                }
                .padding(.horizontal)

                // Topic Selector Chips
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(TeacherTopic.allCases) { topic in
                            let selected = (viewModel.teacherTopic == topic)
                            Button {
                                viewModel.selectTeacherTopic(topic)
                            } label: {
                                Label(topic.title, systemImage: topic.iconName)
                                    .font(.caption.weight(.semibold))
                                    .padding(.horizontal, 11)
                                    .padding(.vertical, 7)
                                    .foregroundStyle(selected ? .white : .secondary)
                                    .background(
                                        selected ? Color.indigo : Color.white.opacity(0.08),
                                        in: Capsule()
                                    )
                            }
                            .disabled(viewModel.teacherIsBusy || viewModel.teacherIsListening)
                        }
                    }
                    .padding(.horizontal)
                }

                if let err = viewModel.teacherErrorMessage {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                // Main Interactive Lesson Card
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        if viewModel.teacherIsBusy {
                            HStack(spacing: 10) {
                                ProgressView()
                                    .tint(Color.amberColor)
                                Text("Teacher is preparing your \(targetLanguage.displayName) lesson...")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 28)
                        } else if let lesson = viewModel.teacherCurrentLesson {
                            if !viewModel.teacherLastLearnerInput.isEmpty {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("YOU SAID (\(knownLanguage.displayName.uppercased()))")
                                        .font(.caption2.weight(.bold))
                                        .foregroundStyle(.secondary)
                                    Text(viewModel.teacherLastLearnerInput)
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(.cyan)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Color.cyan.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
                            }

                            // Teacher prompt in English + user's first language
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(spacing: 8) {
                                    Label("AI Teacher", systemImage: "person.wave.2.fill")
                                        .font(.caption.weight(.bold))
                                        .foregroundStyle(Color.amberColor)
                                    if viewModel.teacherIsSpeaking {
                                        Text("Speaking...")
                                            .font(.caption2.weight(.semibold))
                                            .foregroundStyle(.cyan)
                                            .padding(.horizontal, 7)
                                            .padding(.vertical, 2)
                                            .background(Color.cyan.opacity(0.16), in: Capsule())
                                    }
                                }
                                Text(lesson.teacherPromptText)
                                    .font(.subheadline)
                                    .foregroundStyle(.white.opacity(0.95))
                                if let secondary = lesson.teacherGuidanceSecondary, !secondary.isEmpty {
                                    Text(secondary)
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(Color.amberColor.opacity(0.95))
                                }
                            }

                            // Bilingual Starter / Translation Box + Audio Listen controls
                            VStack(alignment: .leading, spacing: 10) {
                                if !lesson.sourceNativeScript.isEmpty {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("IN YOUR LANGUAGE (\(knownLanguage.displayName.uppercased()))")
                                            .font(.caption2.weight(.bold))
                                            .foregroundStyle(Color.amberColor)
                                        Text(lesson.sourceNativeScript)
                                            .font(.headline.weight(.semibold))
                                            .foregroundStyle(.white)
                                        if knownLanguage != .english,
                                           targetLanguage != .english,
                                           !lesson.englishMeaning.isEmpty {
                                            Text("Meaning: \(lesson.englishMeaning)")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    Divider()
                                        .overlay(Color.white.opacity(0.18))
                                }

                                Text("TRANSLATION & SPEAK IN \(targetLanguage.displayName.uppercased())")
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(.mint)

                                Text(lesson.targetNativeScript)
                                    .font(.title3.weight(.semibold))
                                    .foregroundStyle(.white)

                                if lesson.targetPhoneticSpeak.lowercased() != lesson.targetNativeScript.lowercased() {
                                    Text("Say: \(lesson.targetPhoneticSpeak)")
                                        .font(.subheadline)
                                        .foregroundStyle(.cyan)
                                }

                                if !lesson.grammarTip.isEmpty {
                                    Text("💡 \(lesson.grammarTip)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                HStack(spacing: 10) {
                                    Button {
                                        viewModel.replayTeacherAudio(slow: false)
                                    } label: {
                                        Label("Listen", systemImage: "speaker.wave.2.fill")
                                            .font(.caption.weight(.semibold))
                                            .frame(maxWidth: .infinity)
                                            .padding(.vertical, 8)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(.cyan)
                                    .disabled(viewModel.teacherIsListening)

                                    Button {
                                        viewModel.replayTeacherAudio(slow: true)
                                    } label: {
                                        Label("Slow (0.78x)", systemImage: "tortoise.fill")
                                            .font(.caption.weight(.semibold))
                                            .frame(maxWidth: .infinity)
                                            .padding(.vertical, 8)
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.mint)
                                    .disabled(viewModel.teacherIsListening)
                                }
                            }
                            .padding(14)
                            .background(Color.indigo.opacity(0.24), in: RoundedRectangle(cornerRadius: 14))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(Color.mint.opacity(0.28), lineWidth: 1)
                            )

                            // Pronunciation Score Card (after user practices speaking)
                            if let eval = viewModel.teacherLastEvaluation {
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(eval.matchPercentage >= 75 ? "🌟 Great Pronunciation!" : "🎯 Practice Feedback")
                                            .font(.subheadline.weight(.bold))
                                            .foregroundStyle(eval.matchPercentage >= 75 ? Color.mint : Color.amberColor)
                                        Spacer()
                                        Text("\(eval.matchPercentage)%")
                                            .font(.headline.monospacedDigit())
                                            .foregroundStyle(eval.matchPercentage >= 75 ? Color.mint : Color.amberColor)
                                    }
                                    Text(eval.feedbackMessage)
                                        .font(.caption)
                                        .foregroundStyle(.white)
                                    Text("Heard: \"\(eval.spokenPhonetic)\"")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                .padding(12)
                                .background(
                                    (eval.matchPercentage >= 75 ? Color.green : Color.orange).opacity(0.16),
                                    in: RoundedRectangle(cornerRadius: 12)
                                )
                            }
                        }
                    }
                    .padding(.horizontal)
                }

                if viewModel.teacherIsListening {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 10, height: 10)
                        Text(viewModel.teacherListeningLabel)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white)
                    }
                    .padding(.vertical, 6)
                }

                // Interactive Teacher Controls
                VStack(spacing: 8) {
                    HStack(spacing: 10) {
                        Button {
                            viewModel.toggleTeacherMic(practicingTarget: false)
                        } label: {
                            Label(
                                viewModel.teacherIsListening ? "Stop Mic" : "Ask in \(knownLanguage.displayName)",
                                systemImage: viewModel.teacherIsListening ? "stop.fill" : "mic.fill"
                            )
                            .font(.caption.weight(.semibold))
                            .lineLimit(1)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(viewModel.teacherIsListening ? .red : .indigo)
                        .disabled(viewModel.teacherIsBusy)

                        Button {
                            viewModel.toggleTeacherMic(practicingTarget: true)
                        } label: {
                            Label(
                                viewModel.teacherIsListening ? "Stop Mic" : "Practice \(targetLanguage.displayName)",
                                systemImage: "waveform.badge.mic"
                            )
                            .font(.caption.weight(.semibold))
                            .lineLimit(1)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(viewModel.teacherIsListening ? .red : .mint)
                        .disabled(viewModel.teacherIsBusy || viewModel.teacherCurrentLesson == nil)
                    }

                    HStack(spacing: 10) {
                        Button {
                            viewModel.cancelTeacherSpeech()
                            showKnownKeyboardSheet = true
                        } label: {
                            Label("Type in \(knownLanguage.displayName)", systemImage: "keyboard")
                                .font(.caption.weight(.semibold))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.bordered)
                        .tint(.cyan)
                        .disabled(viewModel.teacherIsBusy || viewModel.teacherIsListening)

                        Button {
                            viewModel.nextTeacherChallenge()
                        } label: {
                            Label("Next Lesson", systemImage: "arrow.right.circle.fill")
                                .font(.caption.weight(.semibold))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.bordered)
                        .tint(Color.amberColor)
                        .disabled(viewModel.teacherIsBusy || viewModel.teacherIsListening)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
            .padding(.top, 8)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.05, green: 0.07, blue: 0.13), Color(red: 0.09, green: 0.11, blue: 0.19)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
            )
            .navigationTitle("Language Teacher")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(isPresented: $showKnownKeyboardSheet) {
                TypeInputSheet(
                    inputLanguage: knownLanguage,
                    outputLanguage: targetLanguage,
                    interactionMode: .translate
                ) { typed in
                    viewModel.sendTeacherKnownText(typed)
                }
            }
        }
    }
}
