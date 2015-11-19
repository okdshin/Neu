include_directories(../)
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
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

add_neu_test("softmax_loss" test_softmax_loss.cpp)
add_neu_test("softmax_loss_layer" test_softmax_loss_layer.cpp)
#add_neu_test("max_pooling_layer" test_max_pooling_layer.cpp)
#add_neu_test("average_pooling_layer" test_average_pooling_layer.cpp)
#add_neu_test("local_responce_normalization_layer" test_local_responce_normalization_layer.cpp)
#add_neu_test("convolution_layer" test_convolution_layer.cpp)
#add_neu_test("convolution_indices" test_convolution_indices.cpp)
#add_neu_test("convolution" test_convolution.cpp)
#add_neu_test("general_cvonvolution" test_general_convolution.cpp)
#add_neu_test("data_set" test_data_set.cpp)
#add_neu_test("load_cifar10" test_load_cifar10.cpp)
#add_neu_test("load_mnist" test_load_mnist.cpp)
#add_neu_test("activation_func" test_activation_func.cpp)
#add_neu_test("yaml_io" test_yaml_io.cpp)
#add_neu_test("compute_test" test_compute_test.cpp)
#add_neu_test("layer" test_layer.cpp)
#add_neu_test("gpu_buffer_range" test_gpu_buffer_range.cpp)
#add_neu_test("fully_connected_layer" test_fully_connected_layer.cpp)
#add_neu_test("activation_layer" test_activation_layer.cpp)
