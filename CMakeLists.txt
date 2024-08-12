
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

project(cartpole_sim)

include(FetchContent)  

set(mujoco_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/include/mujoco)
set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/bin)

find_package(glfw3 REQUIRED)
find_library(CPPAD_LIB cppad_lib)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)

include_directories(
    ./
    ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/include
    simulate
    controllers/iLQR
    ${Python3_INCLUDE_DIRS} 
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Fetch lodepng dependency.
if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG ${MUJOCO_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()

aux_source_directory(simulate SIM_SRC)
aux_source_directory(controllers/iLQR ILQR_SRC)

add_executable(main main.cpp
                    ${ILQR_SRC}
                    ${SIM_SRC}

)

target_link_libraries(main
    glfw
    ${CPPAD_LIB}
    ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/lib/libmujoco.so.3.1.0
    lodepng
    yaml-cpp::yaml-cpp
    Python3::Python 
    Python3::NumPy
)

