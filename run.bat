@echo off
chcp 65001 > nul
title منظومة التعرف الذكي على الأرقام (1-9)
color 0B

echo ======================================================================
echo       🚀 تشغيل منظومة التعرف الذكي على الأرقام المكتوبة بخط اليد
echo ======================================================================
echo.

set PYTHON_CMD=python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Users\admin\AppData\Local\Python\bin\python.exe" (
        set PYTHON_CMD="C:\Users\admin\AppData\Local\Python\bin\python.exe"
    ) else if exist "C:\Users\admin\AppData\Local\Programs\Python\Python314\python.exe" (
        set PYTHON_CMD="C:\Users\admin\AppData\Local\Programs\Python\Python314\python.exe"
    ) else (
        echo [خطأ] لم يتم العثور على مفسر بايثون في جهازك.
        pause
        exit /b 1
    )
)

echo [1/3] جاري التحقق من النموذج المُدرّب...
if not exist "%~dp0models\mnist_random_forest_model.pkl" (
    if exist "%~dp0..\mnist_random_forest_model.pkl" (
        echo جاري نسخ النموذج من المجلد الرئيسي...
        copy "%~dp0..\mnist_random_forest_model.pkl" "%~dp0models\mnist_random_forest_model.pkl" > nul
    )
)

echo [2/3] جاري إطلاق خادم الذكاء الاصطناعي على المنفذ 8080...
start "" http://localhost:8080

echo [3/3] تم فتح التطبيق في المتصفح الافتراضي.
echo اضغط Ctrl+C لإيقاف الخادم عند الانتهاء.
echo.

%PYTHON_CMD% "%~dp0server.py" 8080
pause
