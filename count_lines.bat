@echo off
setlocal

REM 切换代码页到 UTF-8 以防止中文乱码
chcp 65001 >nul

REM --- 配置 ---
REM 设置要统计的文件或目录列表
set TARGETS=CMakeLists.txt core\ main.cpp tests\ ui\

REM 设置排除列表文件
set EXCLUDE_FILE=cloc_exclude.txt

REM 初始化 cloc 命令变量
set CLOC_CMD=

REM 检测 cloc 是否在 PATH 环境变量中
where /q cloc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set CLOC_CMD=cloc
) else (
    REM 检查当前目录下是否有 cloc-*.exe (简单查找第一个)
    for %%F in (cloc-*.exe) do (
        if not defined CLOC_CMD (
             set CLOC_CMD=%%F
        )
    )
)

REM 检查 cloc 命令是否找到
if "%CLOC_CMD%"=="" (
    echo 错误: 未找到 'cloc' 命令或 'cloc-*.exe' 文件。
    echo 请确保 cloc 已安装或 cloc-*.exe 存在于当前目录。
    pause
    exit /b 1
)

REM --- 检查排除文件是否存在 ---
set EXCLUDE_ARG=
if exist "%EXCLUDE_FILE%" (
    set EXCLUDE_ARG=--exclude-list-file="%EXCLUDE_FILE%"
    echo 排除列表文件: %EXCLUDE_FILE%
    REM --- 显示排除列表内容 ---
    echo --------------------
    type "%EXCLUDE_FILE%"
    echo --------------------
) else (
    echo 注意: 排除列表文件 '%EXCLUDE_FILE%' 不存在。将不排除任何文件。
)

REM --- 执行 ---
echo.
echo 正在使用 %CLOC_CMD% 统计代码行数...
echo 目标: %TARGETS%
echo ----------------------------------------

REM 调用 cloc 并传入目标列表和排除参数
%CLOC_CMD% %EXCLUDE_ARG% %TARGETS%

@REM echo.
@REM pause