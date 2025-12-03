# CMakeList.txt: PoseBridge 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.10)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("PoseBridge")

set(CMAKE_CXX_STANDARD 17)

# 查找包 (假设使用 vcpkg)
find_package(OpenCV REQUIRED)
find_package(cppzmq REQUIRED)
find_package(imgui CONFIG REQUIRED) 
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# 将源代码添加到此项目的可执行文件。
add_executable (PoseBridge WIN32  "src/main.cpp")

target_link_libraries(PoseBridge PRIVATE 
    opencv_core opencv_highgui opencv_imgproc opencv_videoio
    cppzmq
    imgui::imgui 
    glfw
    OpenGL::GL
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PoseBridge PROPERTY CXX_STANDARD 20)
endif()

if(MSVC)
    set_target_properties(PoseBridge PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

# TODO: 如有需要，请添加测试并安装目标。
