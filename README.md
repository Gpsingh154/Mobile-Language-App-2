# Vaani Local — 100% Offline Indic Voice-to-Voice iOS App

A native SwiftUI iOS application running an end-to-end offline voice assistant on your iPhone, pre-configured for **Windows 11 + iTunes + iCloud + Sideloadly** deployment:

1. **Audio Input:** `AVAudioEngine` -> `AVAudioConverter` -> `16,000 Hz, 16-bit, Mono PCM`
2. **ASR (Listening):** Sherpa-ONNX (`Silero VAD` + `IndicConformer INT8 ONNX`) outputting native regional scripts (Gurmukhi, Devanagari, Tamil, Bengali, etc.)
3. **Local Brain (LLM):** Google `Gemma 4 E2B-IT` (`.litertlm`) via `LiteRT-LM` on Apple Metal GPU, constrained to output Romanised phonetic prose
4. **TTS (Speaking):** `Kokoro-82M` (INT8/FP16 ONNX) via ONNX Runtime Swift API synthesizing 24 kHz audio played gaplessly through `AVAudioPlayerNode`

---

## Windows 11 + Sideloadly Deployment Workflow (No Mac Required)

Windows 11 cannot directly compile ARM64 iOS binaries because Apple's `xcodebuild` and iOS SDK only run on macOS. Instead, this repo includes a free **GitHub Actions macOS Cloud Builder** that outputs an unsigned `.ipa` you can flash straight onto your iPhone using **Sideloadly + iTunes + iCloud** on Windows 11.

### Step 1: Build the Unsigned `.ipa` via GitHub Actions

1. Push this folder to a GitHub repository (`.gitignore` already excludes multi-GB model files and temporary build output).
2. In your GitHub repository, go to **Actions -> Build Unsigned IPA for Sideloadly -> Run workflow** (or push to `main`).
3. When the `macos-15` job completes, download the **`VaaniLocal-Sideloadly-IPA`** artifact (`VaaniLocal-Sideloadly.ipa`) onto your Windows 11 PC.

> **Why entitlements are Sideloadly-ready:** `OfflineIndicVoiceAI/Resources/OfflineIndicVoiceAI.entitlements` omits paid-only kernel entitlements (`increased-memory-limit`, `extended-virtual-addressing`) because Sideloadly with a free Apple ID rejects restricted entitlements during provisioning. The app uses `ResidencyPolicy.adaptiveLowMemory` by default to manage RAM safely under standard iOS Jetsam limits. (An optional `OfflineIndicVoiceAI-PaidDev.entitlements` is provided if you later sign with a paid Apple Developer certificate or AltStore/SideStore v2.2+.)

### Step 2: Sideload onto Your iPhone on Windows 11

1. Make sure **iTunes** (non-Microsoft-Store version) and **iCloud for Windows** (non-Microsoft-Store version) are installed and running, as required by Sideloadly.
2. Connect your iPhone to your Windows 11 PC via USB and tap **Trust This Computer** on the iPhone.
3. Open **Sideloadly**, drag **`VaaniLocal-Sideloadly.ipa`** into the IPA box, enter your Apple ID, and click **Start**.
4. On your iPhone:
   - Trust the developer profile under **Settings -> General -> VPN & Device Management**.
   - Enable **Settings -> Privacy & Security -> Developer Mode** if prompted (iOS 16+), and restart the iPhone.

### Step 3: Transfer the ~2 GB Offline Model Files (Without Bloating the `.ipa`)

Keeping weights outside the `.ipa` prevents Sideloadly USB timeouts and preserves your models across 7-day Sideloadly refreshes (just re-sideload over the existing app without deleting it first).

**Method A — iTunes File Sharing on Windows 11 (Fastest via USB):**
1. Place your 7 downloaded model files in a folder named `RawModels\` inside this project and run:
   ```powershell
   .\scripts\prepare_models_folder.ps1
   ```
2. Open **iTunes** on Windows 11 -> click the **iPhone icon** -> **File Sharing** -> select **Vaani Local**.
3. Drag the 7 staged files from `StagedModelsForITunes\` into **Vaani Local Documents**:
   - `indicconformer_int8.onnx`
   - `tokens.txt`
   - `silero_vad.onnx`
   - `gemma-4-E2B-it.litertlm`
   - `kokoro-82m-q8.onnx`
   - `af_bella.bin`
   - `vocab.json`
4. Open **Vaani Local** on your iPhone — `ModelAssetStore` automatically detects files placed in `Documents/`.

**Method B — iCloud Drive on Windows 11:**
1. Copy the 7 model files into **iCloud Drive** from Windows 11.
2. Open **Vaani Local -> Models** on your iPhone and tap **Import File** next to each asset.

**Method C — Direct Wi-Fi Download inside the App:**
- Open **Vaani Local -> Models -> Download URL** to fetch weights directly onto your iPhone over HTTPS.

---

## Linking `SherpaOnnx` (Optional Live ASR)

`project.yml` links `LiteRTLM` and `onnxruntime` (`OnnxRuntimeBindings`) out of the box. Because `k2-fsa/sherpa-onnx` bundles `csukuangfj/onnxruntime-libs`, linking both SPM packages simultaneously causes duplicate `OrtGetApiBase` symbols unless deduplicated:
- Without `SherpaOnnx` linked, the app builds cleanly and lets you test the UI, `AVAudioEngine` 16 kHz capture, and the mock dry-run button (**Run Sample Punjabi Turn (Mock)**).
- To enable live on-device IndicConformer ASR, attach `sherpa-onnx.xcframework` + `SherpaOnnx.swift` + `SherpaOnnx-Bridging-Header.h` (reusing a single `onnxruntime.xcframework`) and define `-DSHERPA_ONNX_ENABLED` in Swift compiler flags.
