cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MCCNN)
set (PREF "/home/mohamedali/Documents/Evaluation_MCC-CNN_CODE/MC_CNN_CPP_PYTORCH/libtorch")
list(APPEND CMAKE_FIND_ROOT_PATH "/home/mohamedali/Documents/Evaluation_MCC-CNN_CODE/MC_CNN_CPP_PYTORCH")
list(APPEND CMAKE_PREFIX_PATH "${PREF}")
find_package(Torch REQUIRED)
SET(OpenCV_DIR "/home/mohamedali/miniconda3/envs/TorchEnv/lib")
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${PREF}/..)
set (OpenCV_INCLUDE_DIRS "/home/mohamedali/miniconda3/envs/TorchEnv/include/opencv4")
set (OPENCVPREFIX "/home/mohamedali/miniconda3/envs/TorchEnv/lib")
#set (OpenCV_LIBS "/home/mohamedali/miniconda3/envs/TorchEnv/lib/")
set (Torch_INCLUDE_DIRS "${PREF}/libtorch/include" "${PREF}/libtorch/include/torch/csrc/api/include")
#set (TORCH_LIBRARIES "${PREF}/libtorch/lib")
include_directories(${OpenCV_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS})
add_executable(TestMainCpuGpu TestMainCpuGpu.cpp cv.cpp)
target_link_libraries(TestMainCpuGpu ${TORCH_LIBRARIES})
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_core.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_cvv.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_calib3d.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_highgui.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_imgcodecs.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_features2d.so)
target_link_libraries(TestMainCpuGpu ${OPENCVPREFIX}/libopencv_imgproc.so)
set_property(TARGET TestMainCpuGpu PROPERTY CXX_STANDARD 14)




# The following code block is suggested to be used on Windows.

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TestMainCpuGpu>)
endif (MSVC)
