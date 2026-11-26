//
//  ModelAssetStore.swift
//  OfflineIndicVoiceAI
//
//  Manages offline ASR, LLM, and TTS model assets on iOS using
//  Application Support (excluded from iCloud backups) with support
//  for Bundle fallback, Files app import, and resumable Wi-Fi downloads.
//

import Combine
import Foundation
import os

public enum ModelAssetKind: String, CaseIterable, Identifiable, Codable, Sendable {
    case indicConformerONNX
    case indicConformerTokens
    case sileroVAD
    case gemma4LiteRTLM
    case kokoro82mONNX
    case kokoroVoicePack
    case kokoroVocabJSON

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .indicConformerONNX: return "IndicConformer INT8 ASR"
        case .indicConformerTokens: return "ASR Vocabulary (tokens.txt)"
        case .sileroVAD: return "Silero VAD v4/v5"
        case .gemma4LiteRTLM: return "Gemma 4 E2B-IT (.litertlm)"
        case .kokoro82mONNX: return "Kokoro 82M (INT8/FP16 ONNX)"
        case .kokoroVoicePack: return "Kokoro Voice Style (.bin)"
        case .kokoroVocabJSON: return "Kokoro Phoneme Vocab (vocab.json)"
        }
    }

    public var defaultFileName: String {
        switch self {
        case .indicConformerONNX: return "indicconformer_int8.onnx"
        case .indicConformerTokens: return "tokens.txt"
        case .sileroVAD: return "silero_vad.onnx"
        case .gemma4LiteRTLM: return "gemma-4-E2B-it.litertlm"
        case .kokoro82mONNX: return "kokoro-82m-q8.onnx"
        case .kokoroVoicePack: return "af_bella.bin"
        case .kokoroVocabJSON: return "vocab.json"
        }
    }

    public var expectedSizeDescription: String {
        switch self {
        case .indicConformerONNX: return "~188 - 650 MB"
        case .indicConformerTokens: return "< 1 MB"
        case .sileroVAD: return "~2 MB"
        case .gemma4LiteRTLM: return "~1.3 - 1.8 GB"
        case .kokoro82mONNX: return "~90 - 165 MB"
        case .kokoroVoicePack: return "~2 MB"
        case .kokoroVocabJSON: return "< 1 MB"
        }
    }

    public var allowedFileExtensions: [String] {
        switch self {
        case .indicConformerONNX, .sileroVAD, .kokoro82mONNX: return ["onnx"]
        case .indicConformerTokens: return ["txt"]
        case .gemma4LiteRTLM: return ["litertlm", "task", "bin"]
        case .kokoroVoicePack: return ["bin"]
        case .kokoroVocabJSON: return ["json"]
        }
    }
}

public struct ModelAssetStatus: Identifiable, Sendable {
    public let kind: ModelAssetKind
    public let resolvedURL: URL?
    public let byteSize: Int64
    public let isBundled: Bool

    public var id: ModelAssetKind { kind }
    public var isInstalled: Bool { resolvedURL != nil }

    public var formattedSize: String {
        guard byteSize > 0 else { return "Not installed" }
        return ByteCountFormatter.string(fromByteCount: byteSize, countStyle: .file)
    }
}

@MainActor
public final class ModelAssetStore: NSObject, ObservableObject {
    @Published public private(set) var statuses: [ModelAssetKind: ModelAssetStatus] = [:]
    @Published public private(set) var downloadProgress: [ModelAssetKind: Double] = [:]
    @Published public private(set) var activeDownloadKind: ModelAssetKind?
    @Published public private(set) var availableDiskSpaceGB: Double = 0
    @Published public var lastAssetError: String?

    // Persisted runtime settings
    @Published public var audioRoutingModeRaw: String {
        didSet { UserDefaults.standard.set(audioRoutingModeRaw, forKey: Keys.routingMode) }
    }
    @Published public var residencyPolicyRaw: String {
        didSet { UserDefaults.standard.set(residencyPolicyRaw, forKey: Keys.residencyPolicy) }
    }
    @Published public var ttsSpeed: Double {
        didSet { UserDefaults.standard.set(ttsSpeed, forKey: Keys.ttsSpeed) }
    }
    @Published public var useCoreMLForKokoro: Bool {
        didSet { UserDefaults.standard.set(useCoreMLForKokoro, forKey: Keys.kokoroCoreML) }
    }
    @Published public var selectedVoiceID: String {
        didSet { UserDefaults.standard.set(selectedVoiceID, forKey: Keys.voiceID) }
    }

    private enum Keys {
        static let routingMode = "offlineindic.routingMode"
        static let residencyPolicy = "offlineindic.residencyPolicy"
        static let ttsSpeed = "offlineindic.ttsSpeed"
        static let kokoroCoreML = "offlineindic.kokoroCoreML"
        static let voiceID = "offlineindic.voiceID"
    }

    private let logger = Logger(subsystem: "com.offlineindic.voice", category: "ModelAssetStore")
    private let fileManager = FileManager.default
    private var downloadSession: URLSession?
    private var activeTask: URLSessionDownloadTask?

    public lazy var modelsDirectoryURL: URL = {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        let dir = base.appendingPathComponent("OfflineIndicModels", isDirectory: true)
        ensureExcludedDirectoryExists(at: dir)
        return dir
    }()

    public override init() {
        let defaults = UserDefaults.standard
        self.audioRoutingModeRaw = defaults.string(forKey: Keys.routingMode) ?? "fullDuplex"
        self.residencyPolicyRaw = defaults.string(forKey: Keys.residencyPolicy) ?? "adaptive"
        let savedSpeed = defaults.double(forKey: Keys.ttsSpeed)
        self.ttsSpeed = savedSpeed > 0 ? savedSpeed : 1.0
        self.useCoreMLForKokoro = defaults.bool(forKey: Keys.kokoroCoreML)
        self.selectedVoiceID = defaults.string(forKey: Keys.voiceID) ?? "af_bella"

        super.init()
        refreshStatuses()
    }

    public var allRequiredAssetsInstalled: Bool {
        ModelAssetKind.allCases.allSatisfy { statuses[$0]?.isInstalled == true }
    }

    public var audioRoutingMode: VoicePipelineConfiguration.AudioRoutingMode {
        audioRoutingModeRaw == "halfDuplex" ? .halfDuplexGated : .fullDuplexVoiceChatAEC
    }

    public var residencyPolicy: VoicePipelineConfiguration.ResidencyPolicy {
        switch residencyPolicyRaw {
        case "always": return .alwaysResident
        case "strict": return .strictForegroundOnly
        default: return .adaptiveLowMemory
        }
    }

    public func refreshStatuses() {
        var next: [ModelAssetKind: ModelAssetStatus] = [:]
        for kind in ModelAssetKind.allCases {
            next[kind] = resolveStatus(for: kind)
        }
        statuses = next
        availableDiskSpaceGB = Self.queryAvailableStorageGB(at: modelsDirectoryURL)
    }

    public func buildPipelineConfiguration() -> VoicePipelineConfiguration {
        let pathFor: (ModelAssetKind) -> String = { [weak self] kind in
            guard let self else { return "" }
            return self.statuses[kind]?.resolvedURL?.path
                ?? self.modelsDirectoryURL.appendingPathComponent(kind.defaultFileName).path
        }

        let asr = VoicePipelineConfiguration.ASRConfig(
            indicConformerModelPath: pathFor(.indicConformerONNX),
            tokensPath: pathFor(.indicConformerTokens),
            sileroVadModelPath: pathFor(.sileroVAD)
        )
        let llm = VoicePipelineConfiguration.LLMConfig(
            modelPath: pathFor(.gemma4LiteRTLM)
        )
        let tts = VoicePipelineConfiguration.TTSConfig(
            kokoroModelPath: pathFor(.kokoro82mONNX),
            voicePackPath: pathFor(.kokoroVoicePack),
            vocabJSONPath: pathFor(.kokoroVocabJSON),
            defaultVoiceID: selectedVoiceID,
            speed: Float(ttsSpeed),
            useCoreMLExecutionProvider: useCoreMLForKokoro
        )
        return VoicePipelineConfiguration(
            asr: asr,
            llm: llm,
            tts: tts,
            residencyPolicy: residencyPolicy,
            audioRoutingMode: audioRoutingMode
        )
    }

    /// Imports a local model file selected from the iOS Files app (or AirDrop) into
    /// Application Support with iCloud backup exclusion enabled.
    public func importModelFile(from sourceURL: URL, as kind: ModelAssetKind) {
        let accessed = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if accessed { sourceURL.stopAccessingSecurityScopedResource() }
        }

        do {
            ensureExcludedDirectoryExists(at: modelsDirectoryURL)
            let targetName = (kind == .kokoroVoicePack) ? "\(selectedVoiceID).bin" : kind.defaultFileName
            let destination = modelsDirectoryURL.appendingPathComponent(targetName)
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: sourceURL, to: destination)
            markExcludedFromBackup(url: destination)
            lastAssetError = nil
            refreshStatuses()
        } catch {
            logger.error("Model import failed for \(kind.rawValue): \(error.localizedDescription)")
            lastAssetError = "Import failed: \(error.localizedDescription)"
        }
    }

    /// Downloads a model artifact over HTTPS directly to Application Support.
    public func startDownload(from remoteURL: URL, for kind: ModelAssetKind) {
        cancelActiveDownload()
        lastAssetError = nil
        activeDownloadKind = kind
        downloadProgress[kind] = 0.0

        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        let session = URLSession(configuration: config, delegate: self, delegateQueue: .main)
        self.downloadSession = session

        let task = session.downloadTask(with: remoteURL)
        task.taskDescription = kind.rawValue
        self.activeTask = task
        task.resume()
    }

    public func cancelActiveDownload() {
        activeTask?.cancel()
        activeTask = nil
        downloadSession?.invalidateAndCancel()
        downloadSession = nil
        if let activeDownloadKind {
            downloadProgress[activeDownloadKind] = nil
        }
        activeDownloadKind = nil
    }

    public func deleteAsset(_ kind: ModelAssetKind) {
        let names = (kind == .kokoroVoicePack)
            ? ["\(selectedVoiceID).bin", kind.defaultFileName]
            : [kind.defaultFileName]
        for name in names {
            try? fileManager.removeItem(at: modelsDirectoryURL.appendingPathComponent(name))
            if let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
                try? fileManager.removeItem(at: docsDir.appendingPathComponent(name))
            }
        }
        refreshStatuses()
    }

    public func deleteAllDownloadedModels() {
        cancelActiveDownload()
        for kind in ModelAssetKind.allCases {
            deleteAsset(kind)
        }
        refreshStatuses()
    }

    private func resolveStatus(for kind: ModelAssetKind) -> ModelAssetStatus {
        let candidateFileNames: [String] = (kind == .kokoroVoicePack)
            ? ["\(selectedVoiceID).bin", kind.defaultFileName]
            : [kind.defaultFileName]

        var searchDirectories: [URL] = [modelsDirectoryURL]
        if let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            searchDirectories.append(docsDir)
        }

        for dir in searchDirectories {
            for fileName in candidateFileNames {
                let candidateURL = dir.appendingPathComponent(fileName)
                if fileManager.fileExists(atPath: candidateURL.path) {
                    markExcludedFromBackup(url: candidateURL)
                    return ModelAssetStatus(
                        kind: kind,
                        resolvedURL: candidateURL,
                        byteSize: fileSize(at: candidateURL),
                        isBundled: false
                    )
                }
            }
        }

        for fileName in candidateFileNames {
            let fileBase = (fileName as NSString).deletingPathExtension
            let fileExt = (fileName as NSString).pathExtension
            if let bundleURL = Bundle.main.url(forResource: fileBase, withExtension: fileExt) {
                return ModelAssetStatus(
                    kind: kind,
                    resolvedURL: bundleURL,
                    byteSize: fileSize(at: bundleURL),
                    isBundled: true
                )
            }
        }

        return ModelAssetStatus(kind: kind, resolvedURL: nil, byteSize: 0, isBundled: false)
    }

    private func fileSize(at url: URL) -> Int64 {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
    }

    private func ensureExcludedDirectoryExists(at url: URL) {
        if !fileManager.fileExists(atPath: url.path) {
            try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        }
        markExcludedFromBackup(url: url)
    }

    private func markExcludedFromBackup(url: URL) {
        var mutableURL = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutableURL.setResourceValues(values)
    }

    private static func queryAvailableStorageGB(at url: URL) -> Double {
        guard let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let bytes = values.volumeAvailableCapacityForImportantUsage else {
            return 0
        }
        return Double(bytes) / 1_000_000_000.0
    }
}

extension ModelAssetStore: URLSessionDownloadDelegate {
    nonisolated public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0,
              let rawKind = downloadTask.taskDescription,
              let kind = ModelAssetKind(rawValue: rawKind) else { return }

        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        Task { @MainActor [weak self] in
            self?.downloadProgress[kind] = min(1.0, max(0.0, progress))
        }
    }

    nonisolated public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let rawKind = downloadTask.taskDescription,
              let kind = ModelAssetKind(rawValue: rawKind) else { return }

        // Move synchronously out of URLSession's temporary location before returning.
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fm.temporaryDirectory
        let modelsDir = appSupport.appendingPathComponent("OfflineIndicModels", isDirectory: true)
        try? fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let destination = modelsDir.appendingPathComponent(kind.defaultFileName)

        do {
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: location, to: destination)
            var mutableDest = destination
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? mutableDest.setResourceValues(values)

            Task { @MainActor [weak self] in
                self?.downloadProgress[kind] = nil
                self?.activeDownloadKind = nil
                self?.refreshStatuses()
            }
        } catch {
            Task { @MainActor [weak self] in
                self?.lastAssetError = "Failed saving \(kind.displayName): \(error.localizedDescription)"
                self?.downloadProgress[kind] = nil
                self?.activeDownloadKind = nil
            }
        }
    }

    nonisolated public func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        if (error as NSError).code == NSURLErrorCancelled { return }
        Task { @MainActor [weak self] in
            self?.lastAssetError = "Download failed: \(error.localizedDescription)"
            if let raw = task.taskDescription, let kind = ModelAssetKind(rawValue: raw) {
                self?.downloadProgress[kind] = nil
            }
            self?.activeDownloadKind = nil
        }
    }
}
