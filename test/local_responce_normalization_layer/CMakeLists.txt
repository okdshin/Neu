add_executable( test_local_responce_normalization_layer main.cpp )
target_link_libraries( test_local_responce_normalization_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( test_local_responce_normalization_layer PROPERTIES OUTPUT_NAME "a.out" )
