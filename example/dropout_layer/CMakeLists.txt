add_executable( dropout_layer main.cpp )
include_directories(${YAMLCPP_INCLUDE_DIR})
target_link_libraries( dropout_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( dropout_layer PROPERTIES OUTPUT_NAME "a.out" )
