param(
    [Parameter(Mandatory = $true)]
    [string]$BuildDirectory,
    [Parameter(Mandatory = $true)]
    [ValidateSet("cuda-rtx30-40-50", "cpu-portable", "cpu-avx2")]
    [string]$Variant,
    [string]$Version = "v1.1.1",
    [string]$OutputDirectory = "artifacts"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDirectory))
$outputPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$packageName = "crisperwhisper-$Version-windows-x64-$Variant"
$stagingParent = Join-Path $buildPath "release-staging"
$stagingPath = Join-Path $stagingParent $packageName

if (-not $buildPath.StartsWith(
    [IO.Path]::GetFullPath($projectRoot),
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "BuildDirectory must resolve inside the repository."
}

$executableCandidates = @(
    (Join-Path $buildPath "bin\Release\crisper-whisper.exe"),
    (Join-Path $buildPath "bin\crisper-whisper.exe")
)
$executable = $executableCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $executable) {
    throw "crisper-whisper.exe was not found in $buildPath"
}

if (Test-Path -LiteralPath $stagingPath) {
    $resolvedStaging = [IO.Path]::GetFullPath($stagingPath)
    $resolvedParent = [IO.Path]::GetFullPath($stagingParent)
    if (-not $resolvedStaging.StartsWith(
        $resolvedParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove an unexpected staging path."
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}

New-Item -ItemType Directory -Path $stagingPath -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingPath "samples") -Force |
    Out-Null
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

Copy-Item -LiteralPath $executable -Destination $stagingPath
foreach ($file in @(
    "README.md",
    "CPP.md",
    "BUILD_LINUX.md",
    "LICENSE",
    "crisperwhispercpp.png"
)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $file) -Destination $stagingPath
}
foreach ($file in @("jfk.wav", "README.md", "LICENSE.whisper.cpp")) {
    Copy-Item `
        -LiteralPath (Join-Path $projectRoot "samples\$file") `
        -Destination (Join-Path $stagingPath "samples")
}

# Bundle the redistributable MSVC/OpenMP runtime so the ZIP works on a clean
# Windows installation without asking users to install another package.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Installer's vswhere.exe was not found."
}
$visualStudio = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$redistRoot = Join-Path $visualStudio "VC\Redist\MSVC"
$redistVersion = Get-ChildItem -LiteralPath $redistRoot -Directory |
    Where-Object { $_.Name -match '^\d+\.' } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $redistVersion) {
    throw "Visual C++ redistributable files were not found."
}
$x64Redist = Join-Path $redistVersion.FullName "x64"
$runtimeDirectories = Get-ChildItem -LiteralPath $x64Redist -Directory |
    Where-Object {
        $_.Name -match '^Microsoft\.VC\d+\.(CRT|OpenMP)$'
    }
foreach ($runtimeName in @(
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
    "vcomp140.dll"
)) {
    foreach ($directory in $runtimeDirectories) {
        $runtimePath = Join-Path $directory.FullName $runtimeName
        if (Test-Path -LiteralPath $runtimePath) {
            Copy-Item -LiteralPath $runtimePath -Destination $stagingPath
            break
        }
    }
}

if ($Variant -eq "cuda-rtx30-40-50") {
    if (-not $env:CUDA_PATH) {
        throw "CUDA_PATH is required to bundle the CUDA runtime."
    }
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    $cudaX64Bin = Join-Path $cudaBin "x64"
    if (Test-Path -LiteralPath $cudaX64Bin) {
        $cudaBin = $cudaX64Bin
    }
    foreach ($pattern in @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvJitLink_*.dll"
    )) {
        $matches = Get-ChildItem -LiteralPath $cudaBin -Filter $pattern -File
        if (-not $matches) {
            throw "Required CUDA runtime file was not found: $pattern"
        }
        $matches | Copy-Item -Destination $stagingPath
    }
}

$archive = Join-Path $outputPath "$packageName.zip"
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}
Compress-Archive -LiteralPath $stagingPath -DestinationPath $archive

Write-Host "Created: $archive"
