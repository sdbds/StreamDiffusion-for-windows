
Set-Location $PSScriptRoot
.\venv\Scripts\activate

$Env:HF_HOME = "../../../huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:PYTHONPATH = $PSScriptRoot

Set-Location demo/realtime-txt2img/view

if (!(Test-Path -Path "build")) {
    Write-Output  "正在NPM编译，如果失败你需要安装单独NPM包..."

    npm install

    npm run build
}

Set-Location ../server

python.exe main.py