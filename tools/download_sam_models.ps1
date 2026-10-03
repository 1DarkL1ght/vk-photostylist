$DefaultModels = @(
    "mobilesam_fp32",
    "mobilesam_int8",
    "sam_b_fp32",
    "sam_b_int8",
    "sam_l_fp32",
    "sam_l_int8",
    "sam_h_fp32",
    "sam_h_int8"
)

$RepoUrls = @{
    "mobilesam_fp32" = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip"
    "mobilesam_int8" = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629_quant.zip"
    "sam_b_fp32"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64.zip"
    "sam_b_int8"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64_quant.zip"
    "sam_l_fp32"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip"
    "sam_l_int8"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195_quant.zip"
    "sam_h_fp32"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip"
    "sam_h_int8"     = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_h_01ec64_quant.zip"
}

if ($args.Count -gt 0) {
    $ModelArg = $args[0]
    if (-not $RepoUrls.ContainsKey($ModelArg)) {
        Write-Host "Error: Model '$ModelArg' not found in the list of available."
        Write-Host "Available models are: $($RepoUrls.Keys -join ' ')."
        exit 1
    }
    $ModelsToDownload = @($ModelArg)
} else {
    $ModelsToDownload = $DefaultModels
}

foreach ($Model in $ModelsToDownload) {
    $Url = $RepoUrls[$Model]
    $TargetDir = Join-Path "models" $Model
    $ZipFile = "temp_$Model.zip"

    Write-Host " Downloading: $Model"

    if (-not (Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }

    # Используем curl.exe — он в разы быстрее Invoke-WebRequest
    curl.exe -L -o $ZipFile $Url --progress-bar
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0 -or -not (Test-Path $ZipFile) -or (Get-Item $ZipFile).Length -eq 0) {
        Write-Host "Error while downloading $Model."
        if (Test-Path $ZipFile) { Remove-Item $ZipFile -Force }
        continue
    }

    Write-Host "-> Inflating..."
    try {
        Expand-Archive -Path $ZipFile -DestinationPath $TargetDir -Force
    } catch {} finally {
        if (Test-Path $ZipFile) {
            Remove-Item $ZipFile -Force
        }
    }
}
