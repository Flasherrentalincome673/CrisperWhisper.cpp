param(
    [switch]$Cpu,
    [string]$BuildDirectory = "build",
    [string]$Configuration = "Release",
    [string]$CudaArchitectures = "86;89;120",
    [ValidateSet("portable", "balance", "fast")]
    [string]$CpuProfile = "balance"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $projectRoot $BuildDirectory

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio Installer's vswhere.exe was not found."
}
$visualStudio = & $vswhere `
    -latest `
    -version "[17.0,18.0)" `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw "Visual Studio 2022 C++ Build Tools were not found."
}

# Import the native x64 compiler environment into this PowerShell process.
# Ninja lets CMake drive nvcc directly, so the separate CUDA Visual Studio
# extension is not required.
$vsDevCmd = Join-Path $visualStudio "Common7\Tools\VsDevCmd.bat"
$environmentLines = & $env:COMSPEC /d /s /c `
    "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
foreach ($line in $environmentLines) {
    if ($line.StartsWith("=")) {
        continue
    }
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) {
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$ninja = Join-Path $visualStudio `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $ninja)) {
    throw "Visual Studio's bundled ninja.exe was not found."
}
$env:Path = "$(Split-Path -Parent $ninja);$env:Path"

$configure = @(
    "-S", $projectRoot,
    "-B", $buildPath,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCRISPERWHISPER_BUILD_TESTS=ON",
    "-DCRISPERWHISPER_CPU_PROFILE=$CpuProfile"
)

if ($Cpu) {
    $configure += "-DCRISPERWHISPER_CUDA=OFF"
} else {
    # CUDA 13 integrates with the VS 2022 toolset.  A newer Visual Studio can
    # be the system-wide CMake default before NVIDIA supports it, so select the
    # known-compatible generator explicitly for GPU builds.
    $configure += "-DCRISPERWHISPER_CUDA=ON"
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        throw "nvcc is not on PATH. Install CUDA or re-run with -Cpu."
    }
    $configure += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

& cmake @configure
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed."
}

& cmake --build $buildPath --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Native build failed."
}

& ctest --test-dir $buildPath -C $Configuration --output-on-failure
if ($LASTEXITCODE -ne 0) {
    throw "Native tests failed."
}

$executable = Join-Path $buildPath "bin\$Configuration\crisper-whisper.exe"
if (-not (Test-Path $executable)) {
    $executable = Join-Path $buildPath "bin\crisper-whisper.exe"
}
Write-Host "Built ($CpuProfile CPU profile): $executable"
