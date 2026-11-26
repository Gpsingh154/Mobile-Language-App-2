//
//  SettingsView.swift
//  OfflineIndicVoiceAI
//
//  Runtime audio, voice synthesis, and iOS Jetsam memory policy settings.
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var assetStore: ModelAssetStore
    let onSettingsSaved: () -> Void

    @Environment(\.dismiss) private var dismiss

    private let availableVoices = [
        "af_bella", "af_sarah", "am_adam", "bf_emma", "hf_alpha", "hf_beta", "hm_omega"
    ]

    var body: some View {
        NavigationStack {
            Form {
                Section("Audio & Language") {
                    Picker("Routing Mode", selection: $assetStore.audioRoutingModeRaw) {
                        Text("Full-Duplex AEC (Barge-In)").tag("fullDuplex")
                        Text("Half-Duplex Gated (Clean TTS)").tag("halfDuplex")
                    }
                    Picker("Transcript Script", selection: $assetStore.scriptDisplayModeRaw) {
                        Text("Auto-Detect Punjabi (ਗੁਰਮੁਖੀ)").tag("auto")
                        Text("Always Punjabi (ਗੁਰਮੁਖੀ)").tag("pa")
                        Text("Raw ASR (Hindi / Indic / English)").tag("native")
                    }
                }

                Section("Kokoro 82M Synthesis") {
                    Picker("Voice Style Pack", selection: $assetStore.selectedVoiceID) {
                        ForEach(availableVoices, id: \.self) { voice in
                            Text(voice).tag(voice)
                        }
                    }

                    VStack(alignment: .leading) {
                        HStack {
                            Text("Speech Speed")
                            Spacer()
                            Text(String(format: "%.2fx", assetStore.ttsSpeed))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $assetStore.ttsSpeed, in: 0.75...1.35, step: 0.05)
                    }

                    Toggle("Use CoreML Execution Provider", isOn: $assetStore.useCoreMLForKokoro)
                }

                Section {
                    Picker("RAM Residency", selection: $assetStore.residencyPolicyRaw) {
                        Text("Adaptive Low Memory (Recommended)").tag("adaptive")
                        Text("Always Resident (>= 8GB iPhone)").tag("always")
                        Text("Strict Foreground Only").tag("strict")
                    }
                } header: {
                    Text("iOS Jetsam Memory Guard")
                } footer: {
                    Text("Adaptive mode keeps Gemma 4 warm on Metal GPU and dynamically pages Sherpa-ASR / Kokoro-TTS when os_proc_available_memory() drops below 450 MB.")
                }

                Section("Hardware Telemetry") {
                    HStack {
                        Text("Physical Device RAM")
                        Spacer()
                        Text(String(format: "%.2f GB", MemoryBudgetManager.totalPhysicalRAMGB))
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Current Jetsam Headroom")
                        Spacer()
                        Text("\(MemoryBudgetManager.availableMemoryMB()) MB")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Pipeline Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Apply") {
                        onSettingsSaved()
                        dismiss()
                    }
                }
            }
        }
    }
}
