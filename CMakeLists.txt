cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(fbgemm VERSION 0.1 LANGUAGES CXX C)

set(FBGEMM_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (shared, static, or default) to build")
set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)
option(FBGEMM_BUILD_DOCS "Build fbgemm documentation" OFF)
option(FBGEMM_BUILD_FBGEMM_GPU "Build fbgemm_gpu library" OFF)

if(FBGEMM_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMM_THIRDPARTY_DIR ${FBGEMM_BINARY_DIR}/third_party)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#add address sanitizer
set(USE_SANITIZER "" CACHE STRING "options include address, leak, ...")

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

#check if compiler supports openmp
find_package(OpenMP)
if (OpenMP_FOUND)
  message(WARNING "OpenMP found! OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
  include_directories(${OpenMP_C_INCLUDE_DIRS})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP is not supported by the compiler")
endif()

# Define file lists
get_filelist("get_fbgemm_generic_srcs(with_base=True)" FBGEMM_GENERIC_SRCS)
get_filelist("get_fbgemm_lasx_srcs()" FBGEMM_LASX_SRCS)
get_filelist("get_fbgemm_inline_lasx_srcs()"
  FBGEMM_LASX_INLINE_SRCS)
get_filelist("get_fbgemm_public_headers()" FBGEMM_PUBLIC_HEADERS)

add_library(fbgemm_generic OBJECT ${FBGEMM_GENERIC_SRCS})
add_library(fbgemm_lasx OBJECT ${FBGEMM_LASX_SRCS} ${FBGEMM_LASX_INLINE_SRCS})

# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(fbgemm_generic defs.bzl)
add_dependencies(fbgemm_lasx defs.bzl)

set_target_properties(fbgemm_generic fbgemm_lasx PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden
       )

  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  #string(APPEND CMAKE_CXX_FLAGS " -Werror")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")

  string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-uninitialized")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")

  # add_definitions(-DFBGEMM_FP16_FALLBACK_TO_REF_KERNEL)

  target_compile_options(fbgemm_generic PRIVATE
    "-mlsx" "-mlasx")
  target_compile_options(fbgemm_lasx PRIVATE
    "-mlsx" "-mlasx")
  set_source_files_properties(
    src/FbgemmI64.cc
    src/FbgemmI8Depthwise3DLasx.cc
    src/FbgemmI8DepthwiseLasx.cc
    src/UtilsLasx.cc
    PROPERTIES COMPILE_FLAGS "-Wno-uninitialized")
  set_source_files_properties(src/PackMatrix.cc
    PROPERTIES COMPILE_FLAGS "-Wno-infinite-recursion")
  # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
  if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3.0))
    set_source_files_properties(
      src/GenerateKernelU8S8S32ACC32.cc
      src/GenerateKernelDirectConvU8S8S32ACC32.cc
      PROPERTIES COMPILE_FLAGS "-Wno-attributes")
  endif()

if(USE_SANITIZER)
  target_compile_options(fbgemm_generic PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_lasx PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

message(WARNING "==========")
message(WARNING "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(WARNING "FBGEMM_LIBRARY_TYPE = ${FBGEMM_LIBRARY_TYPE}")
message(WARNING "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
message(WARNING "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message(WARNING "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message(WARNING "==========")

if(NOT TARGET asmjit)
  #Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/third_party/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  #build asmjit
  if ((FBGEMM_LIBRARY_TYPE STREQUAL "shared"))
    set(ASMJIT_STATIC OFF)
  else()
    set(ASMJIT_STATIC ON)
  endif()

  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT TARGET cpuinfo)
  #Download cpuinfo from github if CPUINFO_SOURCE_DIR is not specified.
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/cpuinfo"
      CACHE STRING "cpuinfo source directory from submodules")
  endif()

  #build cpuinfo
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  set(CPUINFO_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  set(CPUINFO_RUNTIME_TYPE static CACHE STRING "Set runtime to static")
  add_subdirectory("${CPUINFO_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/cpuinfo")
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(fbgemm_generic BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

target_include_directories(fbgemm_lasx BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

if(FBGEMM_LIBRARY_TYPE STREQUAL "default")
  add_library(fbgemm
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_lasx>)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
  add_library(fbgemm SHARED
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_lasx>)
  set_property(TARGET fbgemm_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fbgemm_lasx PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(fbgemm PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm STATIC
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_lasx>)
  target_compile_definitions(fbgemm_generic PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_lasx PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm PRIVATE FBGEMM_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMM_LIBRARY_TYPE}")
endif()

if(USE_SANITIZER)
  target_link_options(fbgemm PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

target_include_directories(fbgemm BEFORE
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)

target_link_libraries(fbgemm $<BUILD_INTERFACE:asmjit>
  $<BUILD_INTERFACE:cpuinfo>)
add_dependencies(fbgemm asmjit cpuinfo)

install(TARGETS fbgemm EXPORT fbgemmLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

install(FILES ${FBGEMM_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm")

install(EXPORT fbgemmLibraryConfig DESTINATION share/cmake/fbgemm
  FILE fbgemmLibraryConfig.cmake)


  if(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
    install(
      FILES $<TARGET_PDB_FILE:fbgemm> $<TARGET_PDB_FILE:asmjit>
      DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  endif()
  install(TARGETS fbgemm DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS asmjit DESTINATION ${CMAKE_INSTALL_LIBDIR})


#Make project importable from the build directory
#export(TARGETS fbgemm asmjit FILE fbgemmLibraryConfig.cmake)

if(FBGEMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(FBGEMM_BUILD_DOCS)
  add_subdirectory(docs)
endif()

if(FBGEMM_BUILD_FBGEMM_GPU)
  add_subdirectory(fbgemm_gpu)
endif()
