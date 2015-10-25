add_executable( xor_sigmoid main.cpp )
target_link_libraries( xor_sigmoid
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${OpenCL_LIBRARIES}
)
set_target_properties( xor_sigmoid PROPERTIES OUTPUT_NAME "a.out" )
