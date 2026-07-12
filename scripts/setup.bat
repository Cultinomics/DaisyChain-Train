@echo off
REM ============================================================
REM  DaisyChain - Old Hardware Training Pipeline : setup helper
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo.
echo   ____       _            ____ _           _
echo  ^|  _ \ __ _^(_^)___ _   _^/ ___^| ^|__   __ _^(_^)_ __
echo  ^| ^| ^| / _` ^| / __^| ^| ^| ^| ^|   ^| '_ \ / _` ^| ^| '_ \
echo  ^| ^|_^| ^(_^| ^| \__ \ ^|_^| ^| ^|___^| ^| ^| ^| ^(_^| ^| ^| ^| ^| ^|
echo  ^|____/ \__,_^|_^|___/\__, ^|\____^|_^| ^|_^|\__,_^|_^|_^| ^|_^|
echo                     ^|___/   Old Hardware Training Pipeline
echo.
echo  Choose how to run DaisyChain on this machine:
echo    [1] Docker      (recommended on Windows - most reliable)
echo    [2] Python      (native; multi-node gloo is unstable on Windows - use WSL/Linux)
echo    [3] Just install Python deps
echo    [Q] Quit
echo.
set /p choice="  Your choice: "

if /i "%choice%"=="1" goto docker
if /i "%choice%"=="2" goto python
if /i "%choice%"=="3" goto deps
goto end

:docker
where docker >nul 2>nul
if errorlevel 1 (
  echo  [!] Docker not found. Install Docker Desktop from https://docker.com and re-run.
  goto end
)
echo  Building and starting the demo cluster ^(3 nodes + dashboard^)...
docker compose -f docker/docker-compose.yml up --build -d
echo.
echo  [OK] Cluster starting. Opening the dashboard at http://localhost:8080
timeout /t 4 >nul
start "" http://localhost:8080
echo  To stop:  docker compose -f docker/docker-compose.yml down
goto end

:deps
where python >nul 2>nul
if errorlevel 1 ( echo  [!] Python not found. Install Python 3.9+ first. & goto end )
echo  Installing dependencies...
python -m pip install --upgrade pip
python -m pip install torch numpy psutil
python -m pip install -e .
echo  [OK] Installed. You can now run:  daisychain-train
goto end

:python
call :deps
echo.
echo  ---- Configure this node ----
set /p master="  Coordinator IP (MASTER_ADDR, e.g. Tailscale 100.x): "
set /p world="  Total number of machines (WORLD_SIZE): "
set /p rank="  This machine's RANK (0 = coordinator): "
set /p iface="  Network interface (GLOO_SOCKET_IFNAME, e.g. tailscale0): "
set MASTER_ADDR=%master%
set MASTER_PORT=29560
set WORLD_SIZE=%world%
set RANK=%rank%
set GLOO_SOCKET_IFNAME=%iface%
set USE_LIBUV=0
echo.
echo  [!] Note: native multi-node training over gloo is unstable on Windows.
echo      If it hangs, use the Docker option or run the nodes on Linux/WSL.
echo  Launching node RANK=%rank% ...
python -m daisychain.train
goto end

:end
echo.
pause
endlocal
