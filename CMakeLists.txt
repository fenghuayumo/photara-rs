###########################################################################
# Disable insource build

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR
		"CMake generation for Cycles is not allowed within the source directory!"
		"\n Remove the CMakeCache.txt file and try again from another folder, e.g.:"
		"\n "
		"\n rm CMakeCache.txt"
		"\n cd .."
		"\n mkdir cmake-make"
		"\n cd cmake-make"
		"\n cmake ../cycles"
	)
endif()

###########################################################################
# Policies

cmake_minimum_required(VERSION 3.10)

# So library linking is more sane.
if(POLICY CMP0003)
  cmake_policy(SET CMP0003 NEW)
endif()

# So syntax problems are errors.
if(POLICY CMP0010)
  cmake_policy(SET CMP0010 NEW)
endif()

# So BUILDINFO and BLENDERPATH strings are automatically quoted.
if(POLICY CMP0005)
  cmake_policy(SET CMP0005 NEW)
endif()

# Input directories must have CMakeLists.txt
if(POLICY CMP0014)
  cmake_policy(SET CMP0014 NEW)
endif()

# Compile definitions.
if(POLICY CMP0043)
  cmake_policy(SET CMP0043 NEW)
endif()

# Use ROOT variables in find_package.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# install() DESTINATION paths are normalized.
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

# The target_sources() command converts relative paths to absolute.
cmake_policy(SET CMP0076 NEW)

# Avoid having empty buildtype.
set(CMAKE_BUILD_TYPE_INIT "Release")

# Generate compile commands for easy clangd setup.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile commands for clangd")

###########################################################################
# Initialize project.

project(photara_rs)

if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# Ninja's MSVC header scanner must use the same locale as /showIncludes.
# Existing build trees may cache a localized prefix from compiler detection;
# vcvars and build scripts can subsequently select English. Make both the
# compiler invocation and scanner deterministic, retaining any caller launcher.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "Note: including file:")
  foreach(lang C CXX)
    set(_photara_msvc_launcher "${CMAKE_COMMAND}" -E env VSLANG=1033)
    list(APPEND _photara_msvc_launcher ${CMAKE_${lang}_COMPILER_LAUNCHER})
    set(CMAKE_${lang}_COMPILER_LAUNCHER ${_photara_msvc_launcher})
  endforeach()
  unset(_photara_msvc_launcher)
endif()


###########################################################################
# Macros and utilities.

list(APPEND CMAKE_MODULE_PATH
	"${PROJECT_SOURCE_DIR}/src/cmake"
	"${PROJECT_SOURCE_DIR}/src/cmake/Modules"
)

include(compiler_utils)
include(macros)

###########################################################################
# Global settings.

set(CMAKE_INSTALL_MESSAGE LAZY CACHE STRING "Install message verbosity")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "Installation path" FORCE)
endif()

# Effective install path including config folder, as a generator expression.
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(GENERATOR_IS_MULTI_CONFIG)
  string(REPLACE "\${BUILD_TYPE}" "$<CONFIG>" CMAKE_INSTALL_PREFIX_WITH_CONFIG ${CMAKE_INSTALL_PREFIX})
else()
  string(REPLACE "\${BUILD_TYPE}" "" CMAKE_INSTALL_PREFIX_WITH_CONFIG ${CMAKE_INSTALL_PREFIX})
endif()

# Some stuff is different when building standalone Cycles from inside
# Blender source code and when building it from it's own repository.
set(CYCLES_STANDALONE_REPOSITORY TRUE)

# Force standalone build.
set(WITH_CYCLES_STANDALONE TRUE)

# Global compile definitions since add_definitions() adds for all.
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
  $<$<CONFIG:Debug>:DEBUG;_DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
  $<$<CONFIG:MinSizeRel>:NDEBUG>
  $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)

###########################################################################
# Options.

# Use precompiled library dependencies
option(WITH_LIBS_PRECOMPILED        "Use precompiled library dependencies" ON)

# Library dependencies
option(WITH_CYCLES_ALEMBIC          "Build with Alembic procedural support" ON)
option(WITH_CYCLES_EMBREE           "Build with Embree support" ON)
option(WITH_CYCLES_LOGGING          "Build with logging support (using glog)" OFF)
option(WITH_CYCLES_OPENIMAGEDENOISE "Build with OpenImageDenoise support" ON)
option(WITH_CYCLES_OPENSUBDIV       "Build with OpenSubdiv support" ON)
option(WITH_CYCLES_OPENVDB          "Build with OpenVDB support" ON)
option(WITH_CYCLES_PUGIXML          "Build with XML support" ON)
option(WITH_CYCLES_NANOVDB          "Build with NanoVDB support" ON)
option(WITH_CYCLES_OSL              "Build with OSL support" ON)
option(WITH_CYCLES_USD              "Build with USD support" ON)

option(WITH_LEGACY_LIBRARIES        "Build with older VFX platform 2024 libraries, for compatibility with Houdini or USD" OFF)

# Development Features
option(WITH_CYCLES_DEBUG             "Build Cycles with options useful for debugging (e.g., MIS)" OFF)
option(WITH_CYCLES_DEBUG_NAN         "Build Cycles with additional asserts for detecting NaNs and invalid values" OFF)
option(WITH_CYCLES_NATIVE_ONLY       "Build Cycles with native kernel only (which fits current CPU, use for development only)" OFF)
option(WITH_CYCLES_STANDALONE_GUI    "Build Cycles standalone with GUI" OFF)

# NVIDIA CUDA & OptiX
if(NOT APPLE)
  option(WITH_CYCLES_DEVICE_CUDA "Enable Cycles NVIDIA CUDA compute support" ON)
  option(WITH_CYCLES_DEVICE_OPTIX "Enable Cycles NVIDIA OptiX support" ON)
  mark_as_advanced(WITH_CYCLES_DEVICE_CUDA)

  option(WITH_CYCLES_CUDA_BINARIES "Build Cycles NVIDIA CUDA binaries" OFF)
  set(CYCLES_CUDA_BINARIES_ARCH
    sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_61 sm_70 sm_75 sm_86 sm_89 sm_120 compute_75
    CACHE STRING "CUDA architectures to build binaries for"
  )
  option(WITH_CYCLES_CUDA_BUILD_SERIAL "\
Build cubins one after another (useful on machines with limited RAM)"
    OFF
  )
  option(WITH_CUDA_DYNLOAD "\
Dynamically load CUDA libraries at runtime (for developers, makes cuda-gdb work)"
    ON
  )

  set(OPTIX_ROOT_DIR "" CACHE PATH "\
Path to the OptiX SDK root directory, for building Cycles OptiX kernels."
  )
  set(CYCLES_RUNTIME_OPTIX_ROOT_DIR "" CACHE PATH "\
Path to the OptiX SDK root directory. \
When set, this path will be used at runtime to compile OptiX kernels."
  )

  mark_as_advanced(CYCLES_CUDA_BINARIES_ARCH)
  mark_as_advanced(WITH_CYCLES_CUDA_BUILD_SERIAL)
  mark_as_advanced(WITH_CUDA_DYNLOAD)
  mark_as_advanced(OPTIX_ROOT_DIR)
  mark_as_advanced(CYCLES_RUNTIME_OPTIX_ROOT_DIR)
endif()

# Apple Metal (runtime-compiled kernels; CUDA and OptiX are not required).
if(APPLE)
  option(WITH_CYCLES_DEVICE_METAL "Enable Photara Apple Metal compute support" ON)
elseif(WITH_CYCLES_DEVICE_METAL)
  message(FATAL_ERROR "The Metal backend requires macOS")
endif()

# Hydra Render Delegate
option(WITH_CYCLES_HYDRA_RENDER_DELEGATE "Build Cycles Hydra render delegate" ON)

option(WITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD
       "Build Cycles kernels of difference devices in parallel (faster, but uses more memory)" OFF)
mark_as_advanced(WITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD)

# Error Checking
option(WITH_STRICT_BUILD_OPTIONS "When requirements for a build option are not met, error instead of disabling the option" OFF)

# Photara interfaces
option(WITH_DLSS "Enable NVIDIA DLSS Ray Reconstruction" OFF)
option(PHOTARA_BUILD_C_API "Build the Photara C API library" ON)
option(PHOTARA_BUILD_PYTHON "Build the Photara Python module" OFF)

###########################################################################
# Configuration.

include(CheckCXXCompilerFlag)
include(detect_platform)
include(external_libs)
include(dependency_targets)
include(configure_build)
include(CTest)

###########################################################################
# Include directories.

include_directories(
	third_party/cuew/include
	third_party/hipew/include
  third_party/sky/include
  third_party/mikktspace
)

###########################################################################
# Sources.

option(PHOTARA_WITH_OMM "Enable optional NVIDIA OMM CPU baker integration" ON)
set(PHOTARA_OMM_ROOT "" CACHE PATH "OMM SDK source/install root")
set(PHOTARA_OMM_RUNTIME "" CACHE FILEPATH "OMM shared library runtime, if used")
if(PHOTARA_WITH_OMM AND WITH_CYCLES_DEVICE_OPTIX)
  find_path(PHOTARA_OMM_INCLUDE_DIR omm.h
    HINTS "${PHOTARA_OMM_ROOT}/include" "${PHOTARA_OMM_ROOT}/libraries/omm-lib/include")
  find_library(PHOTARA_OMM_LIBRARY NAMES omm-lib omm-sdk
    HINTS "${PHOTARA_OMM_ROOT}/lib" "${PHOTARA_OMM_ROOT}/lib/Release")
  if(PHOTARA_OMM_INCLUDE_DIR AND PHOTARA_OMM_LIBRARY)
    option(PHOTARA_OMM_STATIC "PHOTARA_OMM_LIBRARY is a static SDK archive" OFF)
    include("${CMAKE_SOURCE_DIR}/src/cmake/omm_runtime.cmake")
    photara_validate_omm_runtime()
    add_library(photara_omm_sdk INTERFACE)
    target_include_directories(photara_omm_sdk INTERFACE "${PHOTARA_OMM_INCLUDE_DIR}")
    target_link_libraries(photara_omm_sdk INTERFACE "${PHOTARA_OMM_LIBRARY}")
    # A raw static archive does not carry CMake's transitive link interface.
    # On Windows both static archives and DLL import libraries use .lib, so
    # require an explicit choice rather than guessing from the suffix.
    option(PHOTARA_OMM_STATIC_OPENMP "Static OMM SDK was built with OpenMP" OFF)
    set(PHOTARA_OMM_DEPENDENCY_ROOT "" CACHE PATH "OMM dependency install/build root")
    set(PHOTARA_OMM_EXTRA_LIBRARIES "" CACHE STRING "Additional static OMM link dependencies")
    if(PHOTARA_OMM_STATIC)
      find_library(PHOTARA_OMM_LZ4_LIBRARY NAMES lz4 lz4_static liblz4
        HINTS "${PHOTARA_OMM_ROOT}/lib" "${PHOTARA_OMM_DEPENDENCY_ROOT}/lib"
              "${PHOTARA_OMM_DEPENDENCY_ROOT}/external/lz4/build/cmake"
              "${PHOTARA_OMM_DEPENDENCY_ROOT}/external/lz4/build/cmake/Release")
      find_library(PHOTARA_OMM_XXHASH_LIBRARY NAMES xxhash xxHash libxxhash
        HINTS "${PHOTARA_OMM_ROOT}/lib" "${PHOTARA_OMM_DEPENDENCY_ROOT}/lib"
              "${PHOTARA_OMM_DEPENDENCY_ROOT}/external/xxHash/build/cmake"
              "${PHOTARA_OMM_DEPENDENCY_ROOT}/external/xxHash/build/cmake/Release")
      if(NOT PHOTARA_OMM_LZ4_LIBRARY OR NOT PHOTARA_OMM_XXHASH_LIBRARY)
        message(FATAL_ERROR "Static OMM requires LZ4 and xxHash. Set PHOTARA_OMM_DEPENDENCY_ROOT or PHOTARA_OMM_LZ4_LIBRARY and PHOTARA_OMM_XXHASH_LIBRARY.")
      endif()
      target_link_libraries(photara_omm_sdk INTERFACE
        "${PHOTARA_OMM_LZ4_LIBRARY}" "${PHOTARA_OMM_XXHASH_LIBRARY}"
        ${PHOTARA_OMM_EXTRA_LIBRARIES})
      if(PHOTARA_OMM_STATIC_OPENMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(photara_omm_sdk INTERFACE OpenMP::OpenMP_CXX)
      endif()
    endif()
    target_compile_definitions(photara_omm_sdk INTERFACE WITH_OPTIX_OMM)
    message(STATUS "Photara OMM CPU baker enabled")
  else()
    message(STATUS "OMM SDK not configured; alpha testing retains its original path")
  endif()
endif()

add_subdirectory(src)
add_subdirectory(third_party)

if(PHOTARA_BUILD_C_API OR PHOTARA_BUILD_PYTHON)
  add_subdirectory(photara)
endif()

# Stage the optional SDK runtime for development builds and installation.
if(TARGET photara_omm_sdk AND PHOTARA_OMM_RUNTIME AND NOT PHOTARA_OMM_STATIC)
  add_custom_target(photara_omm_runtime
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/install"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PHOTARA_OMM_RUNTIME}"
            "${CMAKE_SOURCE_DIR}/install")
  foreach(_omm_target photara_python photara_viewer photara_rs)
    if(TARGET ${_omm_target})
      add_dependencies(${_omm_target} photara_omm_runtime)
    endif()
  endforeach()
  install(FILES "${PHOTARA_OMM_RUNTIME}" DESTINATION .)
endif()
