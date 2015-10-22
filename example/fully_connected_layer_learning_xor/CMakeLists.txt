add_executable( fully_connected_layer_learning_xor main.cpp )
target_link_libraries( fully_connected_layer_learning_xor ${OpenCL_LIBRARIES})
set_target_properties( fully_connected_layer_learning_xor PROPERTIES OUTPUT_NAME "a.out" )
