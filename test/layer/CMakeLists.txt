add_executable( test_layer layer.cpp )
target_link_libraries( test_layer ${OpenCL_LIBRARIES})
set_target_properties( test_layer PROPERTIES OUTPUT_NAME "a.out" )
