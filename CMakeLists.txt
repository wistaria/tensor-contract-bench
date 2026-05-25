cmake_minimum_required(VERSION 3.16)

project(tensor_contract_bench VERSION 0.1.0 LANGUAGES CXX)

option(TCB_BUILD_TESTS "Build tensor-contract-bench tests" ON)

add_subdirectory(cpp)

include(CTest)
if(BUILD_TESTING AND TCB_BUILD_TESTS)
  add_subdirectory(tests/cpp)
endif()
