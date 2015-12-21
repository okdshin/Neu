add_executable( convolution_layer main.cpp )
target_link_libraries( convolution_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( convolution_layer PROPERTIES OUTPUT_NAME "a.out" )
