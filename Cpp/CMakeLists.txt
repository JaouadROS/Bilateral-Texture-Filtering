cmake_minimum_required(VERSION 3.0)
project(BilateralTextureFilter)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(BilateralTextureFilter main.cpp)

# Link libraries
target_link_libraries(BilateralTextureFilter ${OpenCV_LIBS})