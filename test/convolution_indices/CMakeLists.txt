add_executable( test_convolution_indices main.cpp )
target_link_libraries( test_convolution_indices
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( test_convolution_indices PROPERTIES OUTPUT_NAME "a.out" )
