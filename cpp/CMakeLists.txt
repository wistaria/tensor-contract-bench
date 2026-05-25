add_library(tcb_core INTERFACE)

target_include_directories(
  tcb_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(tcb_core INTERFACE cxx_std_17)
