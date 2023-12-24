
Set-Location $PSScriptRoot
.\venv\Scripts\activate

$Env:HF_HOME = "../../../huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:PYTHONPATH = $PSScriptRoot

Set-Location demo/realtime-txt2img/view

if (!(Test-Path -Path "build")) {
    Write-Output  "try to download build zip..."

    wget -Uri "https://github.com/sdbds/StreamDiffusion-for-windows/releases/download/1.0/build.zip" -OutFile "./build.zip"

    Expand-Archive -Path ./build.zip -DestinationPath ./build

    Remove-Item -Path ./build.zip -Recurse -Force

}

if (!(Test-Path -Path "build/*")) {
    Write-Output  "unzip or download failed,npm building..."

    npm install

    npm run build
}

Set-Location ../server

python.exe main.py