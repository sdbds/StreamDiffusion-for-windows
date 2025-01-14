Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output  "创建python虚拟环境venv..."
    python -m venv venv
}
.\venv\Scripts\activate

python -m pip install pip==23.0.1 -i https://mirror.baidu.com/pypi/simple

pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple

pip install --no-deps xformers==0.0.23 -i https://mirror.baidu.com/pypi/simple

Write-Output "安装依赖..."

pip install -e .

python setup.py develop easy_install streamdiffusion[tensorrt]

pip install --pre tensorrt --extra-index-url https://pypi.nvidia.com

pip install pywin32 -i https://mirror.baidu.com/pypi/simple

pip install -r examples/screen/requirements.txt -i https://mirror.baidu.com/pypi/simple

pip install -r demo/realtime-txt2img/requirements.txt -i https://mirror.baidu.com/pypi/simple

pip install accelerate -i https://mirror.baidu.com/pypi/simple

Write-Output "安装完毕"
Read-Host | Out-Null ;