cmake_minimum_required(VERSION 3.16)

project(tensor_contract_bench VERSION 0.1.0 LANGUAGES C CXX)

option(TCB_BUILD_TESTS "Build tensor-contract-bench tests" ON)
option(TCB_ENABLE_BLAS "Enable optional C++ BLAS backend" ON)
option(TCB_ENABLE_TBLIS "Enable optional C++ TBLIS backend" OFF)
option(TCB_USE_SYSTEM_YAML_CPP "Use an installed yaml-cpp package instead of FetchContent" OFF)
option(TCB_USE_SYSTEM_NLOHMANN_JSON "Use an installed nlohmann_json package instead of FetchContent" OFF)

add_subdirectory(cpp)

include(CTest)
if(BUILD_TESTING AND TCB_BUILD_TESTS)
  add_subdirectory(tests/cpp)
endif()
