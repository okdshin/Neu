add_executable( xor_sigmoid_node main.cpp )
target_link_libraries( xor_sigmoid_node
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${OpenCL_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( xor_sigmoid_node PROPERTIES OUTPUT_NAME "a.out" )
