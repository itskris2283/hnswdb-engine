cmake_minimum_required(VERSION 3.20)
project(hnswdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HNSWDB_BUILD_TESTS "Build tests" ON)
option(HNSWDB_WERROR "Treat warnings as errors" OFF)
set(HNSWDB_SANITIZE "" CACHE STRING "Comma-separated list of sanitizers (e.g. address,undefined,thread)")
option(HNSWDB_NATIVE "Optimize for native architecture" OFF)

find_package(Threads REQUIRED)

add_library(hnswdb
    src/distance.cpp
    src/flat_index.cpp
    src/hnsw_index.cpp
    src/metadata.cpp
    src/serialization.cpp
    src/thread_pool.cpp
)

target_include_directories(hnswdb PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(MSVC)
    target_compile_options(hnswdb PRIVATE /W4 /MD)
    if(HNSWDB_WERROR)
        target_compile_options(hnswdb PRIVATE /WX)
    endif()
    set_source_files_properties(src/distance.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    target_compile_options(hnswdb PRIVATE -Wall -Wextra -Wpedantic)
    if(HNSWDB_WERROR)
        target_compile_options(hnswdb PRIVATE -Werror)
    endif()
endif()

if(HNSWDB_SANITIZE AND NOT MSVC)
    string(REPLACE "," ";" SANITIZER_LIST "${HNSWDB_SANITIZE}")
    foreach(san IN LISTS SANITIZER_LIST)
        target_compile_options(hnswdb PRIVATE "-fsanitize=${san}")
        target_link_options(hnswdb PRIVATE "-fsanitize=${san}")
    endforeach()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    target_compile_definitions(hnswdb PRIVATE HNSWDB_HAS_AVX2)
endif()

target_link_libraries(hnswdb PRIVATE Threads::Threads)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(pybind11 QUIET)
endif()

if(pybind11_FOUND)
    pybind11_add_module(_core bindings/pybind_module.cpp)
    target_link_libraries(_core PRIVATE hnswdb)
    if(MSVC)
        target_compile_options(_core PRIVATE /W4 /MD)
    else()
        target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    install(TARGETS _core DESTINATION hnswdb)
endif()


if(HNSWDB_BUILD_TESTS)
    include(CTest)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    set(TEST_FILES
        tests/cpp/test_distance.cpp
        tests/cpp/test_flat_index.cpp
        tests/cpp/test_hnsw_basic.cpp
        tests/cpp/test_hnsw_invariants.cpp
        tests/cpp/test_hnsw_delete.cpp
        tests/cpp/test_errors.cpp
        tests/cpp/test_metadata_filter.cpp
        tests/cpp/test_serialization.cpp
        tests/cpp/test_concurrency.cpp
    )

    foreach(TEST_FILE IN LISTS TEST_FILES)
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_FILE})
        target_link_libraries(${TEST_NAME} PRIVATE hnswdb GTest::gtest_main)
        if(MSVC)
            target_compile_options(${TEST_NAME} PRIVATE /W4)
        else()
            target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

