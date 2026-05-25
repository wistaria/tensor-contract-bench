if(TCB_USE_SYSTEM_YAML_CPP)
  find_package(yaml-cpp REQUIRED)
else()
  include(FetchContent)

  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
    URL_HASH SHA256=25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(yaml-cpp)
endif()

if(TCB_USE_SYSTEM_NLOHMANN_JSON)
  find_package(nlohmann_json REQUIRED)
else()
  include(FetchContent)

  set(JSON_BuildTests OFF CACHE INTERNAL "" FORCE)
  set(JSON_Install OFF CACHE INTERNAL "" FORCE)

  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

set(TCB_CORE_SOURCES
  src/case_loader.cpp
  src/reference_matmul.cpp
  src/result_writer.cpp
)

set(TCB_HAVE_BLAS OFF)
if(TCB_ENABLE_BLAS)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    list(APPEND TCB_CORE_SOURCES src/blas_matmul.cpp)
    set(TCB_HAVE_BLAS ON)
  else()
    message(STATUS "BLAS not found; cpp:blas backend and tests will be skipped")
  endif()
endif()
set(TCB_HAVE_BLAS ${TCB_HAVE_BLAS} PARENT_SCOPE)

set(TCB_HAVE_TBLIS OFF)
if(TCB_ENABLE_TBLIS)
  find_path(TCB_TBLIS_INCLUDE_DIR NAMES tblis.h tblis/tblis.h)
  find_library(TCB_TBLIS_LIBRARY NAMES tblis)
  if(TCB_TBLIS_INCLUDE_DIR AND TCB_TBLIS_LIBRARY)
    list(APPEND TCB_CORE_SOURCES src/tblis_matmul.cpp)
    set(TCB_HAVE_TBLIS ON)
  else()
    message(STATUS "TBLIS not found; cpp:tblis backend will be unavailable")
  endif()
endif()
set(TCB_HAVE_TBLIS ${TCB_HAVE_TBLIS} PARENT_SCOPE)

add_library(tcb_core
  ${TCB_CORE_SOURCES}
)

target_include_directories(
  tcb_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(tcb_core PUBLIC cxx_std_17)

if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(tcb_core PUBLIC yaml-cpp::yaml-cpp)
else()
  target_link_libraries(tcb_core PUBLIC yaml-cpp)
endif()

target_link_libraries(tcb_core PRIVATE nlohmann_json::nlohmann_json)

if(TCB_HAVE_BLAS)
  target_compile_definitions(tcb_core PUBLIC TCB_HAVE_BLAS=1)
  if(TARGET BLAS::BLAS)
    target_link_libraries(tcb_core PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(tcb_core PRIVATE ${BLAS_LIBRARIES})
  endif()
endif()

if(TCB_HAVE_TBLIS)
  target_compile_definitions(tcb_core PUBLIC TCB_HAVE_TBLIS=1)
  target_include_directories(tcb_core PRIVATE ${TCB_TBLIS_INCLUDE_DIR})
  target_link_libraries(tcb_core PRIVATE ${TCB_TBLIS_LIBRARY})
endif()

add_executable(tcb-run
  src/tcb_run.cpp
)

target_link_libraries(tcb-run PRIVATE tcb_core)
