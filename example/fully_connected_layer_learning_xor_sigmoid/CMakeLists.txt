add_executable( fully_connected_layer_learning_xor_sigmoid main.cpp )
target_link_libraries( fully_connected_layer_learning_xor_sigmoid ${OpenCL_LIBRARIES})
set_target_properties( fully_connected_layer_learning_xor_sigmoid PROPERTIES OUTPUT_NAME "a.out" )
