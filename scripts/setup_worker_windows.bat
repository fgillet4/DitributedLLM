@echo off
REM Setup script for DistributedLLM worker node on Windows

echo Setting up DistributedLLM worker node for Windows...

REM Get the project root directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd %PROJECT_ROOT%

echo Project root: %PROJECT_ROOT%

REM Create necessary directories
echo Creating directories...
if not exist logs mkdir logs
if not exist models\cache mkdir models\cache
if not exist data mkdir data

REM Check Python version (require 3.9+)
echo Checking Python version...
python --version > temp.txt 2>&1
set /p python_version=<temp.txt
del temp.txt

echo Found %python_version%

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Please ensure Python 3.9+ is installed.
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Check for CUDA support
echo Checking for GPU support...
python -c "import torch; print(torch.cuda.is_available())" > temp.txt 2>&1
set /p has_cuda=<temp.txt
del temp.txt

if "%has_cuda%"=="True" (
    echo GPU support detected.
    set HAS_GPU=true
    
    REM Get GPU info
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
) else (
    echo No GPU support detected. Performance may be limited.
    set HAS_GPU=false
)

REM Check network connectivity
echo Checking network connectivity...
ipconfig | findstr IPv4 || echo Could not determine IP address.

REM Create a default worker config if not exists
if not exist config\worker_config.yaml (
    echo Creating default worker configuration...
    
    REM Get local IP address (this is a basic approach that may not work in all environments)
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
        set local_ip=%%a
        set local_ip=!local_ip:~1!
        goto :ip_found
    )
    :ip_found
    
    REM Create config directory if it doesn't exist
    if not exist config mkdir config
    
    REM Generate a unique worker ID
    for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set datetime=%%a
    set worker_id=worker_win_%COMPUTERNAME%_%datetime:~0,14%
    
    REM Create default worker configuration
    echo # DistributedLLM Worker Configuration> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Worker identification>> config\worker_config.yaml
    echo worker:>> config\worker_config.yaml
    echo   id: "%worker_id%">> config\worker_config.yaml
    echo   host: "%local_ip%">> config\worker_config.yaml
    echo   port: 5556>> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Coordinator connection>> config\worker_config.yaml
    echo coordinator:>> config\worker_config.yaml
    echo   host: "192.168.1.100"  # CHANGE THIS to your coordinator's IP>> config\worker_config.yaml
    echo   port: 5555>> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Resource management>> config\worker_config.yaml
    echo resources:>> config\worker_config.yaml
    echo   max_memory_percent: 80>> config\worker_config.yaml
    echo   max_cpu_percent: 90>> config\worker_config.yaml
    echo   gpu_available: %HAS_GPU%>> config\worker_config.yaml
    
    echo Default worker configuration created. Please edit config\worker_config.yaml to set the correct coordinator IP.
)

REM Create Windows service (optional)
echo If you want to run as a Windows service, you can use NSSM:
echo 1. Download NSSM from https://nssm.cc/
echo 2. Run: nssm install DistributedLLMWorker
echo 3. Set the path to: %PROJECT_ROOT%\venv\Scripts\python.exe
echo 4. Set the arguments to: %PROJECT_ROOT%\src\main.py --mode worker
echo 5. Configure any additional settings as needed

REM Display setup completion message
echo.
echo Worker setup complete.
echo.
echo Before starting the worker, make sure to:
echo 1. Update the coordinator IP in config\worker_config.yaml
echo 2. Ensure the coordinator node is running
echo.
echo To start the worker, run:
echo venv\Scripts\activate
echo python src\main.py --mode worker
echo.
echo If you want to enable auto-discovery, run:
echo python src\main.py --mode worker --discover

REM Return to the original directory
cd %OLDPWD%