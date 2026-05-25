add_executable(tcb_core_types_test core_types_test.cpp)
target_link_libraries(tcb_core_types_test PRIVATE tcb_core)

add_test(NAME tcb_core_types_test COMMAND tcb_core_types_test)

add_executable(tcb_case_loader_test case_loader_test.cpp)
target_link_libraries(tcb_case_loader_test PRIVATE tcb_core)

add_test(NAME tcb_case_loader_test COMMAND tcb_case_loader_test)
set_tests_properties(tcb_case_loader_test PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_executable(tcb_result_writer_test result_writer_test.cpp)
target_link_libraries(tcb_result_writer_test PRIVATE tcb_core)

add_test(NAME tcb_result_writer_test COMMAND tcb_result_writer_test)
set_tests_properties(tcb_result_writer_test PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_executable(tcb_reference_matmul_test reference_matmul_test.cpp)
target_link_libraries(tcb_reference_matmul_test PRIVATE tcb_core)

add_test(NAME tcb_reference_matmul_test COMMAND tcb_reference_matmul_test)
set_tests_properties(tcb_reference_matmul_test PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_executable(tcb_run_test tcb_run_test.cpp)
target_compile_features(tcb_run_test PRIVATE cxx_std_17)

add_test(NAME tcb_run_test COMMAND tcb_run_test $<TARGET_FILE:tcb-run>)
set_tests_properties(tcb_run_test PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
