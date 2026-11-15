# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

set(INC
  ..
)

set(INC_SYS
)

if(WITH_CYCLES_DEVICE_OPTIX OR WITH_CYCLES_DEVICE_CUDA)
  if(NOT WITH_CUDA_DYNLOAD)
    add_definitions(-DCYCLES_CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE}")
  endif()

  add_definitions(-DCYCLES_RUNTIME_OPTIX_ROOT_DIR="${CYCLES_RUNTIME_OPTIX_ROOT_DIR}")
endif()

set(SRC_BASE
  device.cpp
  denoise.cpp
  graphics_interop.cpp
  kernel.cpp
  memory.cpp
  queue.cpp
)

# Retained as the internal host-memory device used by OptiX for denoising and
# large-tile buffer transfers. It is not user-selectable.
set(SRC_CPU
  cpu/device.cpp
  cpu/device.h
  cpu/device_impl.cpp
  cpu/device_impl.h
  cpu/kernel.cpp
  cpu/kernel.h
  cpu/kernel_function.h
)

# Retained because the OptiX device derives from the CUDA device and uses CUDA
# for non-ray-tracing kernels. It is not user-selectable.
set(SRC_CUDA
  cuda/device.cpp
  cuda/device.h
  cuda/device_impl.cpp
  cuda/device_impl.h
  cuda/graphics_interop.cpp
  cuda/graphics_interop.h
  cuda/kernel.cpp
  cuda/kernel.h
  cuda/queue.cpp
  cuda/queue.h
  cuda/util.cpp
  cuda/util.h
)

set(SRC_DUMMY
  dummy/device.cpp
  dummy/device.h
)

set(SRC_METAL
  metal/bvh.h
  metal/bvh.mm
  metal/device.h
  metal/device.mm
  metal/device_impl.h
  metal/device_impl.mm
  metal/graphics_interop.h
  metal/graphics_interop.mm
  metal/kernel.h
  metal/kernel.mm
  metal/queue.h
  metal/queue.mm
  metal/util.h
  metal/util.mm
)

set(SRC_OPTIX
  optix/device.cpp
  optix/device.h
  optix/device_impl.cpp
  optix/device_impl.h
  optix/opacity_micromap.cpp
  optix/queue.cpp
  optix/queue.h
  optix/util.h
)

set(SRC_HEADERS
  device.h
  denoise.h
  graphics_interop.h
  memory.h
  kernel.h
  queue.h
)

set(SRC
  ${SRC_BASE}
  ${SRC_CPU}
  ${SRC_CUDA}
  ${SRC_DUMMY}
  ${SRC_OPTIX}
  ${SRC_HEADERS}
)

set(LIB
  PUBLIC cycles_kernel
  PUBLIC cycles_util
  PRIVATE bf::dependencies::optional::openimagedenoise
  PRIVATE bf::dependencies::optional::osl
)

if(WITH_CYCLES_DEVICE_OPTIX OR WITH_CYCLES_DEVICE_CUDA)
  if(WITH_CUDA_DYNLOAD)
    list(APPEND LIB
      PRIVATE extern_cuew
    )
  else()
    list(APPEND LIB
      PRIVATE ${CUDA_CUDA_LIBRARY}
    )
  endif()
endif()

if(WITH_CYCLES_DEVICE_METAL)
  list(APPEND SRC ${SRC_METAL})
  list(APPEND LIB PRIVATE ${METAL_LIBRARY})
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_device "${LIB}" ${SRC})

source_group("cpu" FILES ${SRC_CPU})
source_group("cuda" FILES ${SRC_CUDA})
source_group("dummy" FILES ${SRC_DUMMY})
source_group("metal" FILES ${SRC_METAL})
source_group("optix" FILES ${SRC_OPTIX})
source_group("common" FILES ${SRC_BASE} ${SRC_HEADERS})

if(TARGET photara_omm_sdk)
  target_link_libraries(cycles_device PRIVATE photara_omm_sdk)
endif()
