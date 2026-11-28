cmake_minimum_required(VERSION 3.24)
if(NOT TARGET strata-split-2 OR NOT SCREAM_ENGINE_DIR STREQUAL "${PROJECT_SOURCE_DIR}/engine")
  message(FATAL_ERROR "The unified build requires the two-card Scream split source (engine/)")
endif()

find_package(CUDAToolkit 12 REQUIRED)
find_package(Threads REQUIRED)
enable_language(C)

# Stage the pinned sources before applying any integration patches.
set(SCREAM_STRATA_SOURCE "" CACHE PATH "existing Strata v0.2.22 checkout (otherwise fetched)")
set(_strata_commit b28ca95e3e6847f2d5e317994f253b66716deeaa)
if(NOT SCREAM_STRATA_SOURCE)
  include(FetchContent)
  FetchContent_Declare(scream_strata
    GIT_REPOSITORY https://github.com/Niko1221/Strata.git
    GIT_TAG v0.2.22
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR _scream_fetch_only
  )
  FetchContent_MakeAvailable(scream_strata)
  set(SCREAM_STRATA_SOURCE "${scream_strata_SOURCE_DIR}")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SCREAM_STRATA_SOURCE}" rev-parse HEAD
  OUTPUT_VARIABLE _strata_head OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT _strata_head STREQUAL _strata_commit)
  message(FATAL_ERROR "Strata v0.2.22 must be at ${_strata_commit}; got ${_strata_head}")
endif()
configure_file("${SCREAM_STRATA_SOURCE}/LICENSE" "${CMAKE_BINARY_DIR}/STRATA-LICENSE" COPYONLY)
find_program(PATCH_EXECUTABLE patch REQUIRED)
function(stage_strata component stage patch_name entrypoint)
  configure_file("${patch_name}" "${patch_name}" COPYONLY)
  set(_patch "${CMAKE_CURRENT_BINARY_DIR}/${patch_name}")
  file(SHA256 "${_patch}" _patch_hash)
  set(_stamp "${stage}/.scream-source")
  set(_stamp_value "${_strata_commit}-${_patch_hash}")
  if(EXISTS "${_stamp}" AND EXISTS "${stage}/${entrypoint}")
    file(READ "${_stamp}" _staged_version)
    if("${_staged_version}" STREQUAL "${_stamp_value}")
      return()
    endif()
  endif()
  set(_archive "${stage}.tar")
  set(_candidate "${stage}-candidate")
  file(REMOVE_RECURSE "${_candidate}")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SCREAM_STRATA_SOURCE}" archive
    --format=tar --output "${_archive}" "${_strata_commit}:${component}"
    COMMAND_ERROR_IS_FATAL ANY)
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_candidate}")
  file(REMOVE "${_archive}")
  execute_process(COMMAND "${PATCH_EXECUTABLE}" -p1 --batch --forward --fuzz=0 -i "${_patch}"
    WORKING_DIRECTORY "${_candidate}" COMMAND_ERROR_IS_FATAL ANY)
  # Preserve unchanged source timestamps when an integration patch changes.
  file(GLOB_RECURSE _staged_files LIST_DIRECTORIES FALSE RELATIVE "${_candidate}" "${_candidate}/*")
  foreach(_file IN LISTS _staged_files)
    get_filename_component(_parent "${stage}/${_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_parent}")
    file(COPY_FILE "${_candidate}/${_file}" "${stage}/${_file}" ONLY_IF_DIFFERENT)
  endforeach()
  file(REMOVE_RECURSE "${_candidate}")
  file(WRITE "${_stamp}" "${_stamp_value}")
endfunction()

set(_ink_rel src/inkling2)
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/strata-source")
stage_strata(engine "${_stage}" strata-integration.patch CMakeLists.txt)
set(_terminal "${CMAKE_CURRENT_BINARY_DIR}/strata-terminal")
stage_strata(terminal "${_terminal}" strata-serving.patch serve/server.py)
configure_file(../engine-inkling2/serve/inkling_tool_stream.py
  "${_terminal}/serve/inkling_tool_stream.py" COPYONLY)
configure_file(runtime_settings.py "${_terminal}/serve/scream_runtime_settings.py" COPYONLY)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/strata-terminal.path" CONTENT "${_terminal}\n")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/scream-launcher.path" CONTENT "${CMAKE_CURRENT_SOURCE_DIR}/scream.py\n")
configure_file(builtin_profiles.json "${CMAKE_BINARY_DIR}/scream-builtin-profiles.json" COPYONLY)
configure_file(settings_reference.json "${CMAKE_BINARY_DIR}/scream-settings-reference.json" COPYONLY)

configure_file(../shared/inkling_sampling.hpp "${_stage}/${_ink_rel}/inkling_sampling.hpp" COPYONLY)
configure_file(../shared/inkling_lookup.hpp "${_stage}/${_ink_rel}/inkling_lookup.hpp" COPYONLY)
configure_file(../shared/inkling_repetition.hpp "${_stage}/${_ink_rel}/inkling_repetition.hpp" COPYONLY)
configure_file(../shared/inference_sampling_fields.hpp "${_stage}/src/program/inference_sampling_fields.hpp" COPYONLY)
configure_file(../shared/inkling_loop.hpp "${_stage}/include/strata/inkling2/loop_schedule.hpp" COPYONLY)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(STRATA_INK_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
option(SCREAM_UNIFIED_UPSTREAM_STRATA "also link Strata's upstream layer-split path" ON)
if(SCREAM_UNIFIED_UPSTREAM_STRATA)
  set(STRATA_BUILD_SPLIT OFF)
  set(STRATA_GGML_DIR "")
  set(STRATA_PORTABLE ON)
  set(STRATA_EXPERIMENTAL_SM60 ON)
  add_subdirectory("${_stage}" strata-upstream EXCLUDE_FROM_ALL)
else()
  add_subdirectory("${_stage}/${_ink_rel}" strata-inkling EXCLUDE_FROM_ALL)
endif()
# The CUDA backend calls the host AVX routines in the Inkling core.
target_link_libraries(strata_inkling_cuda PUBLIC "$<LINK_ONLY:strata_inkling_core>")

add_library(scream_megla STATIC ../engine-inkling2/src/ink2_v1.cu ../engine-inkling2/src/gguf_min.cpp)
target_compile_definitions(scream_megla PRIVATE main=scream_ink2_v1_main)
target_compile_options(scream_megla PRIVATE -O2)
set_target_properties(scream_megla PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(scream_megla PRIVATE CUDA::cublas Threads::Threads)

add_executable(scream-p100 main.cpp)
set_target_properties(scream-p100 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(scream-p100 PRIVATE scream_megla strata-inkling strata-split-2 CUDA::cudart)
if(SCREAM_UNIFIED_UPSTREAM_STRATA)
  target_compile_definitions(scream-p100 PRIVATE SCREAM_HAS_UPSTREAM_STRATA=1)
  target_link_libraries(scream-p100 PRIVATE strata)
endif()

# Publish a runtime tree that can move independently of the build and source trees.
set(_runtime "libexec/scream-p100")
set(_megla "${PROJECT_SOURCE_DIR}/engine-inkling2")
set(_megla_runtime "${_runtime}/engine-inkling2")
set(_licenses "share/licenses/scream-p100")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/install/scream-launcher.path"
  CONTENT "../${_runtime}/unified/scream.py\n")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/install/strata-terminal.path"
  CONTENT "../${_runtime}/strata-terminal\n")
install(TARGETS scream-p100 RUNTIME DESTINATION bin COMPONENT scream-runtime)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/install/scream-launcher.path"
  "${CMAKE_CURRENT_BINARY_DIR}/install/strata-terminal.path"
  DESTINATION bin COMPONENT scream-runtime)
install(FILES scream.py config.py config_names.py runtime_settings.py profile_exchange.py builtin_profiles.json settings_help.py settings_reference.json
  DESTINATION "${_runtime}/unified" COMPONENT scream-runtime)
install(FILES
  "${_megla}/serve/ink2_server.py" "${_megla}/serve/ink2_sampling.py"
  "${_megla}/serve/ink2_vision_device.py" "${_megla}/serve/inkling_tool_stream.py"
  DESTINATION "${_megla_runtime}/serve" COMPONENT scream-runtime)
install(FILES "${_megla}/data/draft_vocab_32768.bin"
  DESTINATION "${_megla_runtime}/data" COMPONENT scream-runtime)
install(DIRECTORY "${_terminal}/" DESTINATION "${_runtime}/strata-terminal" COMPONENT scream-runtime
  PATTERN "__pycache__" EXCLUDE PATTERN ".venv" EXCLUDE PATTERN "*.pyc" EXCLUDE PATTERN "tests" EXCLUDE)
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/NOTICE.md" "${CMAKE_BINARY_DIR}/STRATA-LICENSE"
  DESTINATION "${_licenses}" COMPONENT scream-runtime)
install(FILES "${PROJECT_SOURCE_DIR}/third_party/ggml/LICENSE" DESTINATION "${_licenses}" RENAME GGML-LICENSE
  COMPONENT scream-runtime)
install(FILES README.md CONFIGURATION.md SETTINGS.md settings_reference.json
  DESTINATION share/doc/scream-p100 COMPONENT scream-runtime)
install(FILES "${PROJECT_SOURCE_DIR}/bench/parity.py" "${PROJECT_SOURCE_DIR}/bench/nvidia_vram.py"
  DESTINATION "${_runtime}/bench" COMPONENT scream-runtime)

# These helpers are built independently. Install each set with its own shared libraries.
set(_helpers "${_megla}/build")
set(_helpers_runtime "${_megla_runtime}/build")
install(PROGRAMS "${_helpers}/ink2_tok" "${_helpers}/ink2_vis"
  DESTINATION "${_helpers_runtime}" COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL)
install(PROGRAMS "${_helpers}/llama.cpp/build/bin/llama-tokenize"
  DESTINATION "${_helpers_runtime}/llama.cpp/build/bin" COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL)
install(DIRECTORY "${_helpers}/llama.cpp/build/bin/" DESTINATION "${_helpers_runtime}/llama.cpp/build/bin"
  COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.so*")
install(DIRECTORY "${_helpers}/llama.cpp/gguf-py/gguf" DESTINATION "${_helpers_runtime}/llama.cpp/gguf-py"
  COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
install(FILES "${_helpers}/llama.cpp/LICENSE" DESTINATION "${_licenses}" RENAME LLAMA-LICENSE
  COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL)
install(FILES "${_helpers}/llama.cpp/gguf-py/LICENSE" DESTINATION "${_licenses}" RENAME GGUF-PY-LICENSE
  COMPONENT scream-megla-helpers EXCLUDE_FROM_ALL)
install(PROGRAMS "${_helpers}/ink2_vis_cuda"
  DESTINATION "${_helpers_runtime}" COMPONENT scream-cuda-projector EXCLUDE_FROM_ALL)
install(DIRECTORY "${_helpers}/llama-vision-cuda/bin/" DESTINATION "${_helpers_runtime}/llama-vision-cuda/bin"
  COMPONENT scream-cuda-projector EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.so*")
install(FILES "${_helpers}/llama.cpp/LICENSE" DESTINATION "${_licenses}" RENAME LLAMA-LICENSE
  COMPONENT scream-cuda-projector EXCLUDE_FROM_ALL)

if(STRATA_BUILD_TESTS)
  add_executable(inkling_sampling_test test_inkling_sampling.cpp)
  add_test(NAME inkling_sampling_test COMMAND inkling_sampling_test)
  add_executable(inkling_loop_test test_inkling_loop.cpp)
  add_test(NAME inkling_loop_test COMMAND inkling_loop_test)
  add_executable(inkling_lookup_test test_inkling_lookup.cpp)
  add_test(NAME inkling_lookup_test COMMAND inkling_lookup_test)
  add_executable(native_sampling_fields_test test_native_sampling_fields.cpp)
  add_test(NAME native_sampling_fields_test COMMAND native_sampling_fields_test)
endif()
