@echo off

for %%A in ("%~dp0\..") do set "PROJECT_ROOT=%%~fA"

:: Check build directory.
if not exist "%PROJECT_ROOT%\build\vc14-x64" mkdir "%PROJECT_ROOT%\build\vc14-x64"
pushd "%PROJECT_ROOT%\build\vc14-x64"

:: Edit these variables if you don't wish to use environment ones 
set BOOST_ROOT=%BOOST_ROOT%
set OPENCV_ROOT=%OPENCV_ROOT%
set QT_ROOT=%QT_ROOT%
set VTK_ROOT=%VTK_ROOT%

cmake ^
	-G"Visual Studio 14 Win64" ^
	"-DBOOST_LIBRARYDIR:PATH=%BOOST_ROOT%/x64" ^
	"-DBOOST_ROOT:PATH=%BOOST_ROOT%" ^
	"-DOPENCV_ROOT:PATH=%OPENCV_ROOT%" ^
	"-DQT_ROOT:PATH=%QT_ROOT%" ^
	"-DVTK_ROOT:PATH=%VTK_ROOT%" ^
	../..
if errorlevel 1 goto :error

echo [*] Project generation succeeded!
popd
goto :eof

:error
echo [!] ERROR: Failed to generate project - see above and correct.
popd
exit /b 1
