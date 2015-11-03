add_executable( mnist_conv main.cpp )
target_link_libraries( mnist_conv
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( mnist_conv PROPERTIES OUTPUT_NAME "a.out" )
