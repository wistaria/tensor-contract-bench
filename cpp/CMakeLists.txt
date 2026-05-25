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

add_library(tcb_core
  src/case_loader.cpp
  src/result_writer.cpp
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
