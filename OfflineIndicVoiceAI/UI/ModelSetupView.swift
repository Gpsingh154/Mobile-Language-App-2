//
//  ModelSetupView.swift
//  OfflineIndicVoiceAI
//
//  Allows users to import local .onnx / .litertlm / .bin / .json files from
//  the iOS Files app or download weights directly over Wi-Fi.
//

import SwiftUI
import UniformTypeIdentifiers

struct ModelSetupView: View {
    @ObservedObject var assetStore: ModelAssetStore
    let onAssetsChanged: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var importingKind: ModelAssetKind?
    @State private var showFileImporter = false
    @State private var downloadSheetKind: ModelAssetKind?
    @State private var downloadURLString = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        Label("iPhone Free Storage", systemImage: "internaldrive")
                        Spacer()
                        Text(String(format: "%.1f GB", assetStore.availableDiskSpaceGB))
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Label("Models Status", systemImage: "checkmark.seal")
                        Spacer()
                        Text(assetStore.allRequiredAssetsInstalled ? "All Ready" : "Incomplete")
                            .foregroundStyle(assetStore.allRequiredAssetsInstalled ? .green : .orange)
                    }
                } header: {
                    Text("Device Storage")
                } footer: {
                    Text("Models are stored in Application Support (excluded from iCloud backups). You can also drop files via Finder/Apple Devices using iOS File Sharing.")
                }

                if let error = assetStore.lastAssetError {
                    Section {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                Section("Required Offline Weights") {
                    ForEach(ModelAssetKind.allCases) { kind in
                        assetRow(for: kind)
                    }
                }

                Section {
                    Button(role: .destructive) {
                        assetStore.deleteAllDownloadedModels()
                        onAssetsChanged()
                    } label: {
                        Label("Delete All Local Weights", systemImage: "trash")
                    }
                }
            }
            .navigationTitle("Offline Models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        onAssetsChanged()
                        dismiss()
                    }
                }
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.data, .item],
                allowsMultipleSelection: false
            ) { result in
                guard let importingKind else { return }
                if case .success(let urls) = result, let first = urls.first {
                    assetStore.importModelFile(from: first, as: importingKind)
                    onAssetsChanged()
                }
            }
            .sheet(item: $downloadSheetKind) { kind in
                DownloadURLSheet(
                    kind: kind,
                    urlString: $downloadURLString,
                    onStartDownload: { url in
                        assetStore.startDownload(from: url, for: kind)
                    }
                )
            }
        }
    }

    @ViewBuilder
    private func assetRow(for kind: ModelAssetKind) -> some View {
        let status = assetStore.statuses[kind]
        let progress = assetStore.downloadProgress[kind]

        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: (status?.isInstalled == true) ? "checkmark.circle.fill" : "circle.dashed")
                    .foregroundStyle((status?.isInstalled == true) ? .green : .orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text(kind.displayName)
                        .font(.subheadline.weight(.semibold))
                    Text("\(kind.defaultFileName)  •  Expected \(kind.expectedSizeDescription)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(status?.formattedSize ?? "Missing")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle((status?.isInstalled == true) ? .green : .secondary)
            }

            if let progress {
                HStack {
                    ProgressView(value: progress)
                    Button("Cancel") {
                        assetStore.cancelActiveDownload()
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                }
            } else {
                HStack(spacing: 10) {
                    Button("Import File") {
                        importingKind = kind
                        showFileImporter = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button("Download URL") {
                        downloadURLString = ""
                        downloadSheetKind = kind
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if status?.isInstalled == true && status?.isBundled == false {
                        Spacer()
                        Button(role: .destructive) {
                            assetStore.deleteAsset(kind)
                            onAssetsChanged()
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}

private struct DownloadURLSheet: View {
    let kind: ModelAssetKind
    @Binding var urlString: String
    let onStartDownload: (URL) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Direct HTTPS Weight URL") {
                    TextField("https://huggingface.co/.../\(kind.defaultFileName)", text: $urlString)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                }
                Section {
                    Button("Start Download") {
                        if let url = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)) {
                            onStartDownload(url)
                            dismiss()
                        }
                    }
                    .disabled(URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)) == nil)
                }
            }
            .navigationTitle(kind.displayName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
