@echo off
setlocal ENABLEDELAYEDEXPANSION
title RAG Microservice - Manager

REM Detecta docker-compose
where docker-compose >nul 2>nul
if errorlevel 1 (
  echo [ERRO] docker-compose nao encontrado no PATH.
  echo Instale Docker Desktop ou ajuste o PATH.
  pause
  exit /b 1
)

:menu
cls
echo ===========================================
echo   RAG Microservice - Menu Principal
echo ===========================================
echo 1 ^) Treinar IA
echo 2 ^) Iniciar sistema (subir containers)
echo 3 ^) Parar sistema
echo 4 ^) Recriar indice (limpar vector_store e treinar)
echo 0 ^) Sair
echo.
set /p opt=Sua opcao:

if "%opt%"=="1" goto train
if "%opt%"=="2" goto up
if "%opt%"=="3" goto down
if "%opt%"=="4" goto rebuild_index
if "%opt%"=="0" goto end
goto menu

:choose_stack
echo.
echo Escolha o stack:
echo 1 ^) CPU
echo 2 ^) GPU
set /p stack=Opcao:
if "%stack%"=="1" (set COMPOSE_FILE=docker-compose.cpu.yml) else (
  if "%stack%"=="2" (set COMPOSE_FILE=docker-compose.gpu.yml) else (goto choose_stack)
)
goto :eof

:train
call :choose_stack
echo.
echo 1 ^) Atualizar dados (executa ETL)
echo 2 ^) Voltar
set /p tmode=Opcao:
if "%tmode%"=="1" (
  docker-compose -f %COMPOSE_FILE% run --rm ai_etl bash -lc "python3 -u scripts/etl_build_index.py"
  echo.
  echo [OK] ETL finalizado.
  pause
)
goto menu

:up
call :choose_stack
docker-compose -f %COMPOSE_FILE% up -d
echo.
echo [OK] Containers subindo. A UI deve abrir em http://localhost:8080
start http://localhost:8080
pause
goto menu

:down
call :choose_stack
docker-compose -f %COMPOSE_FILE% down
echo [OK] Containers parados.
pause
goto menu

:rebuild_index
call :choose_stack
echo.
echo Isso vai APAGAR o indice FAISS (volume vector_store) e recriar.
set /p conf=Confirmar? (s/N):
if /I "%conf%" NEQ "S" goto menu

REM tenta achar o volume do vector_store (nome do projeto + sufixo)
for /f "tokens=*" %%V in ('docker volume ls --format "{{.Name}}" ^| findstr /I "vector_store"') do (
  echo Apagando volume: %%V
  docker volume rm %%V >nul 2>nul
)

echo Rodando ETL para recriar o indice...
docker-compose -f %COMPOSE_FILE% run --rm ai_etl bash -lc "python3 -u scripts/etl_build_index.py"
echo [OK] Indice recriado.
pause
goto menu

:end
echo Ate mais!
endlocal
exit /b 0
