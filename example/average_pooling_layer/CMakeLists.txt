add_executable( average_pooling_layer main.cpp )
target_link_libraries( average_pooling_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( average_pooling_layer PROPERTIES OUTPUT_NAME "a.out" )
