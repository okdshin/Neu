include_directories(../)
include_directories(${YAMLCPP_INCLUDE_DIR})
set(BOOST_COMPONENTS unit_test_framework)
set(BOOST_COMPONENTS ${BOOST_COMPONENTS} thread)

find_package(Boost 1.48 REQUIRED COMPONENTS ${BOOST_COMPONENTS})
add_definitions(-DBOOST_TEST_DYN_LINK)
add_definitions(-DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION)

function(add_neu_test TEST_NAME TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET}
	${CMAKE_THREAD_LIBS_INIT}
    ${OpenCL_LIBRARIES}
    ${Boost_LIBRARIES}
	${BoostCompute_LIBRARIES}
	${YAMLCPP_LIBRARY}
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

add_neu_test("layer.any_layer" test_any_layer.cpp)
add_neu_test("layer.average_pooling" test_average_pooling.cpp)
add_neu_test("layer.max_pooling" test_max_pooling.cpp)
#add_neu_test("layer.automatic_differentiation_layer"
#	test_automatic_differentiation_layer.cpp)
