add_executable( local_contrast_normalization_layer main.cpp )
target_link_libraries( local_contrast_normalization_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( local_contrast_normalization_layer PROPERTIES OUTPUT_NAME "a.out" )
