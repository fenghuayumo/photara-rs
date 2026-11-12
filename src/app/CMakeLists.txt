# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

#####################################################################
# Cycles standalone executable
#####################################################################

set(INC
  ..
  "${CMAKE_SOURCE_DIR}/photara/src"
)
set(INC_SYS
)

set(LIB
  cycles_device
  cycles_kernel
  cycles_scene
  cycles_session
  cycles_bvh
  cycles_subd
  cycles_graph
  cycles_util
)

if(WITH_CYCLES_OSL)
  list(APPEND LIB cycles_kernel_osl)
endif()

if(CYCLES_STANDALONE_REPOSITORY)
  list(APPEND LIB extern_sky)
else()
  list(APPEND LIB bf_intern_sky)
endif()

if(WITH_CYCLES_STANDALONE AND WITH_CYCLES_STANDALONE_GUI)
  list(APPEND LIB
    bf::dependencies::epoxy
    bf::dependencies::optional::sdl
  )
endif()

if(WITH_USD)
  # Silence warning from USD headers using deprecated TBB header.
  add_definitions(
    -D__TBB_show_deprecation_message_atomic_H
    -D__TBB_show_deprecation_message_task_H
  )
  list(APPEND LIB
    cycles_hydra
    bf::dependencies::optional::usd
  )
endif()

cycles_external_libraries_append(LIB)

# Common configuration.

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

if(APPLE AND NOT CYCLES_STANDALONE_REPOSITORY)
  set(app_output_dir Blender.app/Contents/MacOS)
  set(app_install_dir Blender.app/Contents/MacOS)
else()
  set(app_install_dir ${CMAKE_INSTALL_PREFIX})
endif()

# Application build targets

if(WITH_CYCLES_STANDALONE)
  set(SRC
    photara_standalone.cpp
    photara_xml.cpp
    photara_xml.h
    oiio_output_driver.cpp
    oiio_output_driver.h
  )

  if(WITH_CYCLES_STANDALONE_GUI)
    list(APPEND SRC
      opengl/display_driver.cpp
      opengl/display_driver.h
      opengl/shader.cpp
      opengl/shader.h
      opengl/window.cpp
      opengl/window.h
    )
  endif()

  add_executable(photara_rs ${SRC} ${INC} ${INC_SYS})
  unset(SRC)

  target_link_libraries(photara_rs PRIVATE ${LIB})

  # Minimal Windows app example, independent of the optional SDL/OpenGL viewer.
  option(PHOTARA_BUILD_VIEWER "Build the Vulkan viewer example (requires glslc)" ON)
  if(WIN32 AND PHOTARA_BUILD_VIEWER)
    add_executable(photara_viewer WIN32 photara_viewer.cpp photara_xml.cpp)
    target_include_directories(photara_viewer SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/lib/windows_x64/vulkan/include")
    target_link_libraries(photara_viewer PRIVATE ${LIB} user32 winmm
      "${CMAKE_SOURCE_DIR}/lib/windows_x64/vulkan/lib/vulkan-1.lib")
    find_program(PHOTARA_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
    foreach(stage vert frag)
      set(shader "${CMAKE_CURRENT_BINARY_DIR}/shaders/viewer.${stage}.spv")
      add_custom_command(OUTPUT "${shader}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${PHOTARA_GLSLC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/viewer.${stage}" -o "${shader}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/viewer.${stage}")
      target_sources(photara_viewer PRIVATE "${shader}")
      install(FILES "${shader}" DESTINATION "${app_install_dir}/shaders")
    endforeach()
    target_link_options(photara_viewer PRIVATE /ENTRY:mainCRTStartup)
    install(TARGETS photara_viewer DESTINATION ${app_install_dir})
    install(FILES "${CMAKE_SOURCE_DIR}/examples/scene_cornell_box.xml"
            DESTINATION "${app_install_dir}/examples")
  endif()


  if(APPLE)
    if(WITH_CYCLES_STANDALONE_GUI)
      # Frameworks used by SDL.
      string(CONCAT _cycles_sdl_frameworks
        " -framework AudioToolbox"
        " -framework AudioUnit"
        " -framework Cocoa"
        " -framework CoreAudio"
        " -framework CoreHaptics"
        " -framework CoreVideo"
        " -framework ForceFeedback"
        " -framework GameController"
      )
      set_property(
        TARGET photara_rs
        APPEND PROPERTY LINK_FLAGS
        "${_cycles_sdl_frameworks}"
      )
    endif()
  endif()

  if(CYCLES_STANDALONE_REPOSITORY)
    cycles_install_libraries(photara_rs)

    if(WITH_USD AND USD_LIBRARY_DIR)
      install(DIRECTORY
        ${USD_LIBRARY_DIR}/usd
        DESTINATION ${CMAKE_INSTALL_PREFIX}
      )

      install(DIRECTORY
        ${USD_LIBRARY_DIR}/../plugin/usd
        DESTINATION ${CMAKE_INSTALL_PREFIX}
      )
    endif()
  endif()

  install(
    TARGETS photara_rs
    DESTINATION ${app_install_dir})

  if(DEFINED app_output_dir)
    set_target_properties(photara_rs
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${app_output_dir}")
  endif()

  add_test(
    NAME photara_rs_version
    COMMAND ${app_install_dir}/$<TARGET_FILE_NAME:photara_rs> --version)
endif()

if(WITH_CYCLES_PRECOMPUTE)
  set(SRC
    photara_precompute.cpp
  )

  add_executable(photara_precompute ${SRC} ${INC} ${INC_SYS})
  unset(SRC)

  target_link_libraries(photara_precompute
    PRIVATE cycles_util
    PRIVATE bf::dependencies::openimageio
    PRIVATE bf::dependencies::optional::tbb
    PRIVATE bf::dependencies::optional::osl)

  if(NOT CYCLES_STANDALONE_REPOSITORY)
    target_link_libraries(photara_precompute
      PRIVATE bf::intern::guardedalloc)
  endif()

  if(DEFINED app_output_dir)
    set_target_properties(photara_precompute
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${app_output_dir}")
  endif()

  install(
    TARGETS photara_precompute
    DESTINATION ${app_install_dir})
endif()
