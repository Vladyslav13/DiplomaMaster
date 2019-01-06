@echo off

:: Build configuration.
set SOLUTION_NAME="Master-Diploma-Vlad.sln"

for %%A in ("%~dp0\..") do set "PROJECT_ROOT=%%~fA"

:: Generate project files.
call "%PROJECT_ROOT%\scripts\cmakegen-msvc14-x64.cmd"
if errorlevel 1 goto :error

:: Check build directory.
if not exist "%PROJECT_ROOT%\build\vc14-x64" mkdir "%PROJECT_ROOT%\build\vc14-x64"
pushd "%PROJECT_ROOT%\build\vc14-x64"

:: Build project.
echo [*] Building Release configuration.
echo ------------------------------------------------------------------------

call "%VS140COMNTOOLS%\vsvars32.bat"
devenv.com %SOLUTION_NAME% /Build Release /Project ALL_BUILD /Out build.log
if errorlevel 1 goto :error

:: Exit
echo [*] Project build succeeded!
popd
goto :eof

:: Error handling
:error
echo [!] ERROR: Failed to build project - see above and correct.
popd
exit /b 1
bat