add_executable( test_activation_func main.cpp )
target_link_libraries( test_activation_func ${OpenCL_LIBRARIES})
set_target_properties( test_activation_func PROPERTIES OUTPUT_NAME "a.out" )
