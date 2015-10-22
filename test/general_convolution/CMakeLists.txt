add_executable( test_general_convolution main.cpp )
target_link_libraries( test_general_convolution
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( test_general_convolution PROPERTIES OUTPUT_NAME "a.out" )
