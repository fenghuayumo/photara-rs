# SPDX-License-Identifier: Apache-2.0

#####################################################################
# Photara C API and Python bindings
#####################################################################

set(PHOTARA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PHOTARA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/photara.cpp")

# Mirror the definitions from src/CMakeLists.txt. add_definitions() is
# directory-scoped, so a sibling directory of src/ does not inherit them.
add_definitions(
  -DCCL_NAMESPACE_BEGIN=namespace\ ccl\ {
  -DCCL_NAMESPACE_END=}
)

if(WITH_OPENIMAGEDENOISE)
  add_definitions(-DWITH_OPENIMAGEDENOISE)
endif()

if(WITH_CYCLES_DEBUG)
  add_definitions(-DWITH_CYCLES_DEBUG)
endif()
if(WITH_CYCLES_OSL)
  add_definitions(-DWITH_OSL)
endif()
if(WITH_CYCLES_DEVICE_CUDA OR WITH_CYCLES_DEVICE_OPTIX)
  add_definitions(-DWITH_CUDA)
  if(WITH_CUDA_DYNLOAD)
    add_definitions(-DWITH_CUDA_DYNLOAD)
  endif()
endif()
if(WITH_CYCLES_DEVICE_OPTIX AND OPTIX_FOUND)
  add_definitions(-DWITH_OPTIX)
endif()

set(PHOTARA_LIB
  cycles_device
  cycles_kernel
  cycles_scene
  cycles_session
  cycles_bvh
  cycles_subd
  cycles_graph
  cycles_util
)

if(CYCLES_STANDALONE_REPOSITORY)
  list(APPEND PHOTARA_LIB extern_sky)
else()
  list(APPEND PHOTARA_LIB bf_intern_sky)
endif()

if(WITH_CYCLES_OSL)
  list(APPEND PHOTARA_LIB cycles_kernel_osl)
endif()

cycles_external_libraries_append(PHOTARA_LIB)

add_library(photara_capi STATIC ${PHOTARA_SRC})
target_include_directories(photara_capi PUBLIC "${PHOTARA_INCLUDE_DIR}")
target_include_directories(photara_capi PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/third_party/atomic")
if(OPTIX_INCLUDE_DIR)
  target_include_directories(photara_capi SYSTEM PRIVATE "${OPTIX_INCLUDE_DIR}")
endif()
target_compile_features(photara_capi PRIVATE cxx_std_17)
target_link_libraries(photara_capi PRIVATE ${PHOTARA_LIB})
if(WIN32)
  target_link_libraries(photara_capi PRIVATE dbghelp)
endif()

if(PHOTARA_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(photara_python python/bindings.cpp)
  target_include_directories(photara_python PRIVATE
    "${PHOTARA_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(photara_python PRIVATE photara_capi)
  target_compile_features(photara_python PRIVATE cxx_std_17)

  set(PHOTARA_PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/photara_rs"
      CACHE PATH "Output directory for the Photara Python extension")
  set_target_properties(photara_python PROPERTIES
    OUTPUT_NAME "_photara"
    RUNTIME_OUTPUT_DIRECTORY "${PHOTARA_PYTHON_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PHOTARA_PYTHON_OUTPUT_DIR}")

  message(STATUS "Photara Python module will be written to ${PHOTARA_PYTHON_OUTPUT_DIR}")
endif()
