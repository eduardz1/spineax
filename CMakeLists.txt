cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Backend options
option(SPINEAX_USE_CUDA "Build with CUDA/cuDSS backend" ON)
option(SPINEAX_USE_BASPACHO "Build with BaSpaCho backend (Metal/OpenCL/CPU)" OFF)

# BaSpaCho path - set this to your baspacho installation
set(BASPACHO_ROOT "" CACHE PATH "Path to BaSpaCho installation")

if(SPINEAX_USE_CUDA)

# Force all CUDA libraries to be linked dynamically.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 1. Find the Python interpreter from your active environment.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python interpreter: ${Python_EXECUTABLE}")

# 2. Use NVIDIA libs from Python site-packages/nvidia/*/ (directories there contain include/ and lib/).
# --- Derive NVIDIA prefix from CMAKE_PREFIX_PATH ---
# We look for a prefix containing "site-packages", then scan nvidia/* for include/lib.

set(_site_prefix "")
foreach(_p IN LISTS CMAKE_PREFIX_PATH)
  message(STATUS "CMAKE_PREFIX_PATH entry: ${_p}")
  # Support both site-packages (pip/venv) and dist-packages (Debian/Ubuntu system Python)
  if(_p MATCHES "site-packages" OR _p MATCHES "dist-packages")
    set(_site_prefix "${_p}")
  endif()
endforeach()

if(NOT _site_prefix)
  message(FATAL_ERROR
    "Could not find a 'site-packages' or 'dist-packages' entry in CMAKE_PREFIX_PATH.\n"
    "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\n")
endif()

set(_nvidia_root "${_site_prefix}/nvidia")

set(ENV_NVIDIA_LIB_DIRS "")
set(ENV_NVIDIA_INCLUDE_DIRS "")
set(ENV_NVIDIA_LEAFS "")

if(EXISTS "${_nvidia_root}")
  file(GLOB _nvidia_candidates LIST_DIRECTORIES true "${_nvidia_root}/*")
  foreach(_cand IN LISTS _nvidia_candidates)
    if(IS_DIRECTORY "${_cand}")
      if(EXISTS "${_cand}/lib")
        list(APPEND ENV_NVIDIA_LIB_DIRS "${_cand}/lib")
        get_filename_component(_leaf "${_cand}" NAME)
        list(APPEND ENV_NVIDIA_LEAFS "${_leaf}")
      endif()
      if(EXISTS "${_cand}/include")
        list(APPEND ENV_NVIDIA_INCLUDE_DIRS "${_cand}/include")
      endif()
    endif()
  endforeach()
endif()

list(REMOVE_DUPLICATES ENV_NVIDIA_LIB_DIRS)
list(REMOVE_DUPLICATES ENV_NVIDIA_INCLUDE_DIRS)
list(REMOVE_DUPLICATES ENV_NVIDIA_LEAFS)

if(NOT ENV_NVIDIA_LIB_DIRS)
  message(FATAL_ERROR
    "Found site-packages prefix: ${_site_prefix}\n"
    "But could not find any nvidia/* with lib/ and include/ under:\n"
    "  ${_nvidia_root}")
endif()

set(ENV_NVIDIA_BUILD_RPATH "")
foreach(_libdir IN LISTS ENV_NVIDIA_LIB_DIRS)
  if(ENV_NVIDIA_BUILD_RPATH)
    set(ENV_NVIDIA_BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}:${_libdir}")
  else()
    set(ENV_NVIDIA_BUILD_RPATH "${_libdir}")
  endif()
endforeach()

set(ENV_NVIDIA_INSTALL_RPATH "")
foreach(_leaf IN LISTS ENV_NVIDIA_LEAFS)
  if(ENV_NVIDIA_INSTALL_RPATH)
    set(ENV_NVIDIA_INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}:$ORIGIN/../nvidia/${_leaf}/lib")
  else()
    set(ENV_NVIDIA_INSTALL_RPATH "$ORIGIN/../nvidia/${_leaf}/lib")
  endif()
endforeach()

message(STATUS "Using NVIDIA lib dirs: ${ENV_NVIDIA_LIB_DIRS}")
foreach(_libdir IN LISTS ENV_NVIDIA_LIB_DIRS)
  message(STATUS "  ${_libdir}")
endforeach()
message(STATUS "Using NVIDIA include dirs:")
foreach(_incdir IN LISTS ENV_NVIDIA_INCLUDE_DIRS)
  message(STATUS "  ${_incdir}")
endforeach()

find_library(ENV_CUDART_PATH
  NAMES cudart libcudart libcudart.so libcudart.so.13 libcudart.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(ENV_CUBLAS_PATH
  NAMES cublas libcublas libcublas.so libcublas.so.13 libcublas.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(ENV_CUDSS_PATH
  NAMES cudss libcudss libcudss.so libcudss.so.0 libcudss.so.13 libcudss.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "Found libcudart at: ${ENV_CUDART_PATH}")
message(STATUS "Found libcudss  at: ${ENV_CUDSS_PATH}")
message(STATUS "Found libcublas at: ${ENV_CUBLAS_PATH}")

# --- Imported targets (clean linking) ---
add_library(nvidia::cudart SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cudart PROPERTIES
  IMPORTED_LOCATION "${ENV_CUDART_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

add_library(nvidia::cudss SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cudss PROPERTIES
  IMPORTED_LOCATION "${ENV_CUDSS_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

add_library(nvidia::cublas SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cublas PROPERTIES
  IMPORTED_LOCATION "${ENV_CUBLAS_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

find_package(nanobind CONFIG REQUIRED)

# XLA
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# 3 optimization build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Option to disable pbatch_solve (useful when nvcc version doesn't match CUDA runtime)
# When OFF, enable_language(CUDA) is not required, so nvcc is not needed.
option(BUILD_PBATCH_SOLVE "Build pseudo batch solve module (requires matching nvcc)" ON)

if(BUILD_PBATCH_SOLVE)
  enable_language(CUDA)
endif()

# cuDSS FFI files

# traditional batch
nanobind_add_module(batch_solve NOMINSIZE
  src/spineax/cudss/batch_solve.cpp
)

# pseudo batch (enable features lacking in normal batch whilst also solving a batch)
if(BUILD_PBATCH_SOLVE)
  nanobind_add_module(pbatch_solve NOMINSIZE
    src/spineax/cudss/pseudo_batch_solve.cu
  )
endif()

# single solve and Return Everything (return all data that cuDSS has)
nanobind_add_module(single_solve_re NOMINSIZE
  src/spineax/cudss/single_solve_re.cpp
)

# traditional single solve
nanobind_add_module(single_solve NOMINSIZE
  src/spineax/cudss/single_solve.cpp
)

# Build list of targets
set(SPINEAX_TARGETS batch_solve single_solve single_solve_re)
if(BUILD_PBATCH_SOLVE)
  list(APPEND SPINEAX_TARGETS pbatch_solve)
endif()

foreach(TARGET ${SPINEAX_TARGETS})
  target_include_directories(${TARGET} PRIVATE
    ${XLA_DIR}
    ${ENV_NVIDIA_INCLUDE_DIRS}
  )

  target_link_libraries(${TARGET} PRIVATE
    nvidia::cudss
    nvidia::cudart
    nvidia::cublas
  )

  # Suppress warnings from XLA FFI headers that have control flow issues
  # These are in jaxlib's XLA headers, not in spineax code
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
  )

  # Add RPATH so the .so can find its dependencies
  # Use $ORIGIN to make paths relative to the installed module location
  # Modules install to site-packages/spineax/, CUDA libs are in site-packages/nvidia/*/lib/
  set_target_properties(${TARGET} PROPERTIES
    BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}"
    INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endforeach()

install(
  TARGETS ${SPINEAX_TARGETS}
  LIBRARY
  DESTINATION spineax
)

#==============================================================================
# BaSpaCho dense solver on CUDA (requires both CUDA and BaSpaCho)
#==============================================================================
if(SPINEAX_USE_BASPACHO AND BASPACHO_ROOT)
  message(STATUS "Building BaSpaCho dense CUDA solver")

  # Find cuSOLVER and cuSPARSE (needed by BaSpaCho's CUDA backend)
  find_library(ENV_CUSOLVER_PATH
    NAMES cusolver libcusolver libcusolver.so libcusolver.so.11
    HINTS ${ENV_NVIDIA_LIB_DIRS}
    NO_DEFAULT_PATH
  )
  find_library(ENV_CUSPARSE_PATH
    NAMES cusparse libcusparse libcusparse.so libcusparse.so.12
    HINTS ${ENV_NVIDIA_LIB_DIRS}
    NO_DEFAULT_PATH
  )

  if(ENV_CUSOLVER_PATH)
    message(STATUS "Found libcusolver at: ${ENV_CUSOLVER_PATH}")
    add_library(nvidia::cusolver SHARED IMPORTED GLOBAL)
    set_target_properties(nvidia::cusolver PROPERTIES
      IMPORTED_LOCATION "${ENV_CUSOLVER_PATH}"
      INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
    )
  endif()
  if(ENV_CUSPARSE_PATH)
    message(STATUS "Found libcusparse at: ${ENV_CUSPARSE_PATH}")
    add_library(nvidia::cusparse SHARED IMPORTED GLOBAL)
    set_target_properties(nvidia::cusparse PROPERTIES
      IMPORTED_LOCATION "${ENV_CUSPARSE_PATH}"
      INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
    )
  endif()

  # BaSpaCho include and library (reuse from BaSpaCho section if already found)
  if(NOT BASPACHO_INCLUDE_DIR)
    set(BASPACHO_INCLUDE_DIR "${BASPACHO_ROOT}")
    set(BASPACHO_BUILD_DIR "${BASPACHO_ROOT}/build")
    set(EIGEN_INCLUDE_DIR "${BASPACHO_BUILD_DIR}/_deps/eigen-src")

    find_library(BASPACHO_LIBRARY
      NAMES BaSpaCho BaSpaCho_static baspacho
      PATHS "${BASPACHO_BUILD_DIR}/baspacho/baspacho" "${BASPACHO_BUILD_DIR}/baspacho" "${BASPACHO_BUILD_DIR}"
      NO_DEFAULT_PATH
    )
    if(NOT BASPACHO_LIBRARY)
      message(FATAL_ERROR "Could not find baspacho library in ${BASPACHO_BUILD_DIR}")
    endif()
    add_library(baspacho STATIC IMPORTED)
    set_target_properties(baspacho PROPERTIES
      IMPORTED_LOCATION "${BASPACHO_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${BASPACHO_INCLUDE_DIR}"
    )
  endif()

  # GPU kernels require nvcc — enable CUDA if not already enabled
  if(NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  endif()

  nanobind_add_module(baspacho_dense_solve NOMINSIZE
    src/spineax/cudss/baspacho_dense_solve.cu
  )

  target_include_directories(baspacho_dense_solve PRIVATE
    ${XLA_DIR}
    ${BASPACHO_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${ENV_NVIDIA_INCLUDE_DIRS}
  )

  target_link_libraries(baspacho_dense_solve PRIVATE
    baspacho
    nvidia::cudart
    nvidia::cublas
  )
  if(ENV_CUSOLVER_PATH)
    target_link_libraries(baspacho_dense_solve PRIVATE nvidia::cusolver)
  endif()
  if(ENV_CUSPARSE_PATH)
    target_link_libraries(baspacho_dense_solve PRIVATE nvidia::cusparse)
  endif()

  # Add dispenso if found
  if(NOT DISPENSO_LIBRARY)
    find_library(DISPENSO_LIBRARY
      NAMES dispenso
      PATHS "${BASPACHO_BUILD_DIR}/_deps/dispenso-build/dispenso"
      NO_DEFAULT_PATH
    )
  endif()
  if(DISPENSO_LIBRARY)
    target_link_libraries(baspacho_dense_solve PRIVATE ${DISPENSO_LIBRARY})
  endif()

  target_compile_options(baspacho_dense_solve PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-return-type --expt-relaxed-constexpr>
  )

  set_target_properties(baspacho_dense_solve PROPERTIES
    BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}"
    INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )

  install(
    TARGETS baspacho_dense_solve
    LIBRARY DESTINATION spineax
  )
endif() # SPINEAX_USE_CUDA AND SPINEAX_USE_BASPACHO

endif() # SPINEAX_USE_CUDA

#==============================================================================
# BaSpaCho backend (Metal/OpenCL/CPU)
#==============================================================================
if(SPINEAX_USE_BASPACHO)
  message(STATUS "Building BaSpaCho backend")

  # Find BaSpaCho
  if(BASPACHO_ROOT)
    set(BASPACHO_INCLUDE_DIR "${BASPACHO_ROOT}")
    set(BASPACHO_BUILD_DIR "${BASPACHO_ROOT}/build")
    set(EIGEN_INCLUDE_DIR "${BASPACHO_BUILD_DIR}/_deps/eigen-src")

    # Find baspacho library (static)
    find_library(BASPACHO_LIBRARY
      NAMES BaSpaCho BaSpaCho_static baspacho
      PATHS "${BASPACHO_BUILD_DIR}/baspacho/baspacho" "${BASPACHO_BUILD_DIR}/baspacho" "${BASPACHO_BUILD_DIR}"
      NO_DEFAULT_PATH
    )

    if(NOT BASPACHO_LIBRARY)
      message(FATAL_ERROR "Could not find baspacho library in ${BASPACHO_BUILD_DIR}")
    endif()

    message(STATUS "Found BaSpaCho: ${BASPACHO_LIBRARY}")
    message(STATUS "BaSpaCho include: ${BASPACHO_INCLUDE_DIR}")

    # Create imported target (static library)
    add_library(baspacho STATIC IMPORTED)
    set_target_properties(baspacho PROPERTIES
      IMPORTED_LOCATION "${BASPACHO_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${BASPACHO_INCLUDE_DIR}"
    )

    # BaSpaCho dependencies - find dispenso
    find_library(DISPENSO_LIBRARY
      NAMES dispenso
      PATHS "${BASPACHO_BUILD_DIR}/_deps/dispenso-build/dispenso"
      NO_DEFAULT_PATH
    )
    if(DISPENSO_LIBRARY)
      message(STATUS "Found dispenso: ${DISPENSO_LIBRARY}")
    endif()
  else()
    message(FATAL_ERROR "BASPACHO_ROOT must be set when SPINEAX_USE_BASPACHO=ON")
  endif()

  # BaSpaCho solve module
  nanobind_add_module(baspacho_solve NOMINSIZE
    src/spineax/cudss/baspacho_solve.cpp
  )

  target_include_directories(baspacho_solve PRIVATE
    ${XLA_DIR}
    ${BASPACHO_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
  )

  # Check for Metal and BLAS support on Apple
  if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
  endif()

  target_link_libraries(baspacho_solve PRIVATE
    baspacho
  )

  # Add dispenso if found
  if(DISPENSO_LIBRARY)
    target_link_libraries(baspacho_solve PRIVATE ${DISPENSO_LIBRARY})
  endif()

  # Add BLAS/LAPACK (Accelerate on macOS)
  if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(baspacho_solve PRIVATE ${ACCELERATE_FRAMEWORK})
  endif()

  # Enable Metal backend on macOS
  if(APPLE AND METAL_FRAMEWORK)
    message(STATUS "Metal framework found - enabling Metal backend")
    target_compile_definitions(baspacho_solve PRIVATE BASPACHO_USE_METAL)
    target_link_libraries(baspacho_solve PRIVATE
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
    )
  endif()

  # Suppress XLA FFI header warnings
  target_compile_options(baspacho_solve PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
  )

  install(
    TARGETS baspacho_solve
    LIBRARY DESTINATION spineax
  )
endif() # SPINEAX_USE_BASPACHO
