cmake_minimum_required(VERSION 3.16)
project(gtcrn_cpp_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# onnxruntime C++ 运行时：从 GitHub releases 下载 win-x64 zip 后解压到此处
# （https://github.com/microsoft/onnxruntime/releases，本仓库验证版本 v1.17.1）
set(ORT_ROOT "D:/data/models/onnxruntime-win/onnxruntime-win-x64-1.17.1" CACHE PATH "onnxruntime 根目录")
find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib" REQUIRED)

add_executable(gtcrn_demo main.cpp)
target_include_directories(gtcrn_demo PRIVATE "${ORT_ROOT}/include")
target_link_libraries(gtcrn_demo PRIVATE ${ORT_LIB})

if(MSVC)
  # 源文件含 UTF-8 中文注释：MSVC 默认按系统代码页解析，必须显式声明 /utf-8
  target_compile_options(gtcrn_demo PRIVATE /O2 /EHsc /utf-8)
else()
  target_compile_options(gtcrn_demo PRIVATE -O2)
endif()
