
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

project(iLQR_sim)

include(FetchContent)  

set(mujoco_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/include/mujoco)
set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/bin)

set(eigen_INCLUDE_DIRS /usr/local/include/eigen3)
set(cppad_INCLUDE_DIRS /usr/local/include/cppad)
set(pinocchio_INCLUDE_DIRS /usr/local/include/pinocchio)
set(blasfeo_DIR /opt/blasfeo)
set(hpipm_DIR /opt/hpipm)

find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(urdfdom REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Python3 COMPONENTS Development)
find_package(Boost REQUIRED COMPONENTS filesystem system)

add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
add_definitions(-DEIEN_USE_THREADING)

include_directories(
    ./
    ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/include
    ${EIGEN3_INCLUDE_DIR}
    ${pinocchio_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${hpipm_DIR}/include
    ${blasfeo_DIR}/include
    simulate
    controllers
    controllers/mpc
    controllers/iLQR
    controllers/idto
    controllers/direct_multiple_shooting
    auto_diff
    examples/common
    examples/common/cost
    examples/common/dynamics
    examples/common/constraint
    examples/cartpole
    examples/quadruped
    ${PYTHON_INCLUDE_DIRS}
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
aux_source_directory(controllers/mpc MPC_SRC)
aux_source_directory(controllers/iLQR ILQR_SRC)
aux_source_directory(controllers/idto IDTO_SRC)
aux_source_directory(controllers/direct_multiple_shooting DMS_SRC)
aux_source_directory(auto_diff AUTO_DIFF_SRC)
aux_source_directory(examples/common COMMON_SRC)
aux_source_directory(examples/common/cost COST_SRC)
aux_source_directory(examples/common/dynamics DYNAMICS_SRC)
aux_source_directory(examples/common/constraint CONSTRAINT_SRC)
aux_source_directory(examples/cartpole CARTPOLE_SRC)
aux_source_directory(examples/quadruped QUADRUPED_SRC)

add_library(shared_lib INTERFACE)
target_link_libraries(shared_lib INTERFACE
  glfw
  ${CMAKE_SOURCE_DIR}/third_party/mujoco-3.1.0/lib/libmujoco.so.3.1.0
  lodepng
  pinocchio
  ${hpipm_DIR}/lib/libhpipm.a
  ${blasfeo_DIR}/lib/libblasfeo.a
  urdfdom_model
  yaml-cpp::yaml-cpp
  ${PYTHON_LIBRARIES}
  ${Boost_LIBRARIES}
  OpenMP::OpenMP_CXX
  Python3::Python
)

add_executable(main main.cpp
                    ${MPC_SRC}
                    ${ILQR_SRC}
                    ${IDTO_SRC}
                    ${DMS_SRC}
                    ${SIM_SRC}
                    ${AUTO_DIFF_SRC}
                    ${COMMON_SRC}
                    ${COST_SRC}
                    ${DYNAMICS_SRC}
                    ${CONSTRAINT_SRC}
                    ${CARTPOLE_SRC}
                    ${QUADRUPED_SRC}
)

target_link_libraries(main
  shared_lib
)


# # test
# add_executable(test_forward_dynamics test/testForwardDynamics.cpp
#   ${AUTO_DIFF_SRC}
#   ${QUADRUPED_SRC}
# )
# target_link_libraries(test_forward_dynamics
#   shared_lib
# )

# add_executable(test_optimizer test/testOptimizer.cpp
#   ${AUTO_DIFF_SRC}
#   ${IDTO_SRC}
# )
# target_link_libraries(test_optimizer
#   shared_lib
# )

add_executable(test_dms test/testMultipleShooting.cpp
  ${MPC_SRC}
  ${DMS_SRC}
  ${AUTO_DIFF_SRC}
  ${COMMON_SRC}
  ${COST_SRC}
  ${DYNAMICS_SRC}
  ${CONSTRAINT_SRC}
  ${QUADRUPED_SRC}
)
target_link_libraries(test_dms
  shared_lib
)






