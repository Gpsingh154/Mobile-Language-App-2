<#
.SYNOPSIS
    Validates and stages the 7 offline AI model files on Windows 11 for easy
    drag-and-drop transfer to your iPhone via iTunes File Sharing or iCloud Drive.
#>
param(
    [string]$SourceFolder = "$PWD\RawModels",
    [string]$OutputFolder = "$PWD\StagedModelsForITunes"
)

$ErrorActionPreference = "Stop"

$expectedFiles = @(
    @{ Name = "indicconformer_int8.onnx"; Purpose = "Sherpa-ONNX IndicConformer INT8 ASR" },
    @{ Name = "tokens.txt";               Purpose = "IndicConformer Vocabulary" },
    @{ Name = "silero_vad.onnx";          Purpose = "Silero Voice Activity Detector" },
    @{ Name = "gemma-4-E2B-it.litertlm";  Purpose = "Gemma 4 E2B-IT LiteRT-LM Model" },
    @{ Name = "kokoro-82m-q8.onnx";       Purpose = "Kokoro 82M Quantized TTS" },
    @{ Name = "af_bella.bin";             Purpose = "Kokoro Voice Style Vector" },
    @{ Name = "vocab.json";               Purpose = "Kokoro Phoneme Vocabulary" }
)

if (-not (Test-Path $SourceFolder)) {
    New-Item -ItemType Directory -Force -Path $SourceFolder | Out-Null
    Write-Host "Created '$SourceFolder'. Place your downloaded model files inside it and re-run this script." -ForegroundColor Yellow
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

$missing = 0
Write-Host "`nChecking model assets in: $SourceFolder`n" -ForegroundColor Cyan

foreach ($item in $expectedFiles) {
    $srcPath = Join-Path $SourceFolder $item.Name
    if (Test-Path $srcPath) {
        $sizeMB = [Math]::Round((Get-Item $srcPath).Length / 1MB, 2)
        Copy-Item -Force $srcPath (Join-Path $OutputFolder $item.Name)
        Write-Host ("[OK]      {0,-28} ({1,8} MB) - {2}" -f $item.Name, $sizeMB, $item.Purpose) -ForegroundColor Green
    } else {
        Write-Host ("[MISSING] {0,-28}              - {1}" -f $item.Name, $item.Purpose) -ForegroundColor Red
        $missing++
    }
}

Write-Host ""
if ($missing -eq 0) {
    Write-Host "All 7 model assets staged in '$OutputFolder'." -ForegroundColor Green
    Write-Host "Next: Connect iPhone -> open iTunes -> iPhone icon -> File Sharing -> Vaani Local -> drag these 7 files in." -ForegroundColor Cyan
} else {
    Write-Host "$missing file(s) missing. Transfer available files or import remaining assets inside the iOS app." -ForegroundColor Yellow
}
