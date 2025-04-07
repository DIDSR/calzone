goto comment

A sketch of the Calzone GUI build workflow

conda create -n calzone-gui pip
pip install nicegui pyinstaller pywin32

also: add build files to .gitignore

pywebview RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize from pythonnet\runtime\Python.Runtime.dll
https://stackoverflow.com/questions/76214672/failed-to-initialize-python-runtime-dll

:comment

rem make a copy of calzone in the gui directory
xcopy "..\calzone" ".\calzone" /E /I
xcopy "../logo.png" ".\calzone" /E /I

rem build
nicegui-pack --onefile --name "Calzone" calzoneGUI.py