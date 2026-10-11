cmake_minimum_required(VERSION 3.16)
project(gtcrn_cpp_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# onnxruntime C++ 运行时：从 GitHub releases 下载 win-x64 zip 后解压到某处
# （https://github.com/microsoft/onnxruntime/releases，本仓库验证版本 v1.17.1）
# V5 加固：ORT_ROOT 不再只有硬编码默认值，优先级为
#   -DORT_ROOT=<路径>  >  环境变量 ONNXRUNTIME_ROOT  >  环境变量 ORT_ROOT  >  仓库默认路径
set(_ORT_DEFAULT "D:/data/models/onnxruntime-win/onnxruntime-win-x64-1.17.1")
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  set(_ORT_DEFAULT "$ENV{ONNXRUNTIME_ROOT}")
elseif(DEFINED ENV{ORT_ROOT})
  set(_ORT_DEFAULT "$ENV{ORT_ROOT}")
endif()
set(ORT_ROOT "${_ORT_DEFAULT}" CACHE PATH
    "onnxruntime 根目录（可 -DORT_ROOT=... 或环境变量 ONNXRUNTIME_ROOT/ORT_ROOT 覆盖）")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ORT_ROOT=${ORT_ROOT} 下找不到 include/onnxruntime_cxx_api.h。\n"
    "请下载 onnxruntime win-x64 并解压，然后用 -DORT_ROOT=<解压目录> 或设置环境变量 ONNXRUNTIME_ROOT 指定。")
endif()
find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib" REQUIRED)

add_executable(gtcrn_demo main.cpp)
target_include_directories(gtcrn_demo PRIVATE "${ORT_ROOT}/include")
target_link_libraries(gtcrn_demo PRIVATE ${ORT_LIB})

# V5 加固：Windows 下把 onnxruntime.dll 部署到可执行文件同目录。
# 旧实现只链接 .lib 不拷 dll —— 运行时靠 PATH/System32 找同名 dll，
# 实测本机 System32 存在 onnxruntime.dll（版本不一定与编译期头文件一致），
# 会静默加载到版本不匹配的运行库。应用目录优先级高于 System32，拷贝后即确定性使用本地版本。
if(WIN32 AND EXISTS "${ORT_ROOT}/lib/onnxruntime.dll")
  add_custom_command(TARGET gtcrn_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:gtcrn_demo>"
    COMMENT "部署 onnxruntime.dll -> $<TARGET_FILE_DIR:gtcrn_demo>")
endif()

if(MSVC)
  # 源文件含 UTF-8 中文注释：MSVC 默认按系统代码页解析，必须显式声明 /utf-8
  target_compile_options(gtcrn_demo PRIVATE /O2 /EHsc /utf-8)
else()
  target_compile_options(gtcrn_demo PRIVATE -O2)
endif()
