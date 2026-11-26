//
//  OfflineIndicVoiceApp.swift
//  OfflineIndicVoiceAI
//
//  App entry point wiring ModelAssetStore and PipelineViewModel into the SwiftUI hierarchy.
//

import SwiftUI

@main
struct OfflineIndicVoiceApp: App {
    @StateObject private var assetStore: ModelAssetStore
    @StateObject private var viewModel: PipelineViewModel

    init() {
        let store = ModelAssetStore()
        _assetStore = StateObject(wrappedValue: store)
        _viewModel = StateObject(wrappedValue: PipelineViewModel(assetStore: store))
    }

    var body: some Scene {
        WindowGroup {
            VoiceAssistantView(viewModel: viewModel, assetStore: assetStore)
                .preferredColorScheme(.dark)
        }
    }
}
