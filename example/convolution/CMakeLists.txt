add_executable( convolution main.cpp )
target_link_libraries( convolution
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( convolution PROPERTIES OUTPUT_NAME "a.out" )
