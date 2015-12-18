add_executable( xor_softmax main.cpp )
target_link_libraries( xor_softmax
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${OpenCL_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( xor_softmax PROPERTIES OUTPUT_NAME "a.out" )
