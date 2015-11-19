add_executable( xor_softmax main.cpp )
target_link_libraries( xor_softmax
	${Boost_LIBRARIES}
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( xor_softmax PROPERTIES OUTPUT_NAME "a.out" )
