@echo off
rem Habilita a expansão de variáveis de ambiente em tempo de execução.
setlocal ENABLEDELAYEDEXPANSION
title RAG Microservice - Manager

REM --- Verificação de Dependências ---
REM Garante que o docker-compose está instalado e acessível no PATH do sistema.
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
echo 1 ^) Treinar IA (executa o ETL para processar os dados)
echo 2 ^) Iniciar sistema (sobe os containers Docker)
echo 3 ^) Parar sistema (para e remove os containers)
echo 4 ^) Recriar indice (apaga o banco de vetores e treina novamente)
echo 0 ^) Sair
echo.
set /p opt=Sua opcao:

rem --- Roteamento das Opções do Menu ---
if "%opt%"=="1" goto train
if "%opt%"=="2" goto up
if "%opt%"=="3" goto down
if "%opt%"=="4" goto rebuild_index
if "%opt%"=="0" goto end
goto menu

rem --- Sub-rotina para Escolher o Ambiente (CPU ou GPU) ---
:choose_stack
echo.
echo Escolha o stack:
echo 1 ^) CPU
echo 2 ^) GPU
set /p stack=Opcao:
rem Define a variável COMPOSE_FILE com base na escolha do usuário.
if "%stack%"=="1" (set COMPOSE_FILE=docker-compose.cpu.yml) else (
  if "%stack%"=="2" (set COMPOSE_FILE=docker-compose.gpu.yml) else (goto choose_stack)
)
goto :eof

rem --- Opção 1: Treinar a IA ---
:train
call :choose_stack
echo.
echo 1 ^) Atualizar dados (executa ETL)
echo 2 ^) Voltar
set /p tmode=Opcao:
if "%tmode%"=="1" (
  rem Executa o container 'ai_etl' de forma intermitente para rodar o script de ETL.
  docker-compose -f %COMPOSE_FILE% run --rm ai_etl bash -lc "python3 -u scripts/etl_build_index.py"
  echo.
  echo [OK] ETL finalizado.
  pause
)
goto menu

rem --- Opção 2: Iniciar o Sistema ---
:up
call :choose_stack
rem Sobe todos os serviços definidos no arquivo docker-compose em modo 'detached' (-d).
docker-compose -f %COMPOSE_FILE% up -d
echo.
echo [OK] Containers subindo. A UI deve abrir em http://localhost:8080
rem Abre o navegador na interface web.
start http://localhost:8080
pause
goto menu

rem --- Opção 3: Parar o Sistema ---
:down
call :choose_stack
rem Para e remove os containers, redes e volumes anônimos.
docker-compose -f %COMPOSE_FILE% down
echo [OK] Containers parados.
pause
goto menu

rem --- Opção 4: Recriar o Índice Vetorial ---
:rebuild_index
call :choose_stack
echo.
echo Isso vai APAGAR o indice FAISS (volume vector_store) e recriar.
set /p conf=Confirmar? (s/N):
rem Garante que a confirmação seja explícita.
if /I "%conf%" NEQ "S" goto menu

rem Procura e remove o volume Docker que armazena o índice vetorial.
rem O nome do volume geralmente é 'nome-do-projeto_vector_store'.
for /f "tokens=*" %%V in ('docker volume ls --format "{{.Name}}" ^| findstr /I "vector_store"') do (
  echo Apagando volume: %%V
  docker volume rm %%V >nul 2>nul
)

rem Após apagar o volume, executa o ETL para criar um novo índice do zero.
echo Rodando ETL para recriar o indice...
docker-compose -f %COMPOSE_FILE% run --rm ai_etl bash -lc "python3 -u scripts/etl_build_index.py"
echo [OK] Indice recriado.
pause
goto menu

:end
echo Ate mais!
endlocal
exit /b 0