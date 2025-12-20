#!/bin/bash

# --- 配置 ---
# 定义要统计的文件和目录列表
TARGETS=("CMakeLists.txt" "core/" "main.cpp" "tests/" "ui/")

# 定义排除列表文件
EXCLUDE_FILE="cloc_exclude.txt"

# 获取 cloc 的路径
# 尝试直接调用 cloc（如果已在PATH中）
if command -v cloc &> /dev/null
then
    CLOC_CMD="cloc"
# 否则，尝试使用你目录下的 cloc-*.exe
# 注意：通配符在变量赋值中可能不会按预期展开，所以最好明确指定或检查
elif ls ./cloc-*.exe 1> /dev/null 2>&1; then
    # 如果只有一个 cloc-*.exe，可以这样获取（简单处理）
    # 更健壮的方法是循环查找或指定确切名称
    CLOC_CMD=(./cloc-*.exe) # This creates an array, take first element
    CLOC_CMD="${CLOC_CMD[0]}"
    if [[ ! -x "$CLOC_CMD" ]]; then
         echo "警告: 找到 $CLOC_CMD 但不可执行。"
         CLOC_CMD=""
    fi
else
    CLOC_CMD=""
fi

# 检查 cloc 命令是否存在
if [[ -z "$CLOC_CMD" ]]; then
   echo "错误: 未找到 'cloc' 命令或 './cloc-<version>.exe' 文件。"
   echo "请确保 cloc 已安装或将 cloc-<version>.exe 放在当前目录。"
   exit 1
fi

# --- 检查排除文件是否存在 ---
if [[ ! -f "$EXCLUDE_FILE" ]]; then
    echo "警告: 排除列表文件 '$EXCLUDE_FILE' 不存在。将不排除任何文件。"
    # 移除 --exclude-list-file 参数
    EXCLUDE_ARG=()
else
    # 构造 cloc 的排除参数
    EXCLUDE_ARG=(--exclude-list-file="$EXCLUDE_FILE")
fi

# --- 执行 ---
echo "正在使用 $CLOC_CMD 统计代码行数..."
echo "目标: ${TARGETS[*]}"
if [[ -f "$EXCLUDE_FILE" ]]; then
    echo "排除列表来自: $EXCLUDE_FILE"
    cat "$EXCLUDE_FILE" | sed 's/^/  /' # Indent list for clarity
fi
echo "----------------------------------------"

# 调用 cloc 并传入目标列表和排除参数
# 使用 "${EXCLUDE_ARG[@]}" 正确展开数组参数
"$CLOC_CMD" "${EXCLUDE_ARG[@]}" "${TARGETS[@]}"