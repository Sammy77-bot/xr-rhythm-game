$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) {
    throw 'Install Node.js 22 or later, then launch the demo again.'
}

$model = Join-Path $PSScriptRoot 'public/keynet.onnx'
$expected = '6AD778F91457CCB71D630FF32C180B43D8D29B4BD9BC1FF9186D762CD95C9C29'
if (-not (Test-Path -LiteralPath $model) -or (Get-FileHash -LiteralPath $model).Hash -ne $expected) {
    $url = 'https://huggingface.co/mattf/pianoCV/resolve/7e7dd57b1a2f77d5f4e29a9e244f7911e9022986/keynet.onnx'
    New-Item -ItemType Directory -Force (Split-Path -Parent $model) | Out-Null
    curl.exe --fail --location --retry 2 --max-time 180 $url --output "$model.download"
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath "$model.download").Hash -ne $expected) {
        throw 'KeyNet download failed or the model checksum did not match.'
    }
    Move-Item -LiteralPath "$model.download" -Destination $model -Force
}

if (-not (Test-Path -LiteralPath 'node_modules/.bin/vite.cmd')) {
    npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
npm.cmd run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
npm.cmd run preview -- --open
exit $LASTEXITCODE
