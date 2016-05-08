add_executable( shared_dropout_layer main.cpp )
include_directories(${YAMLCPP_INCLUDE_DIR})
target_link_libraries( shared_dropout_layer
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( shared_dropout_layer PROPERTIES OUTPUT_NAME "a.out" )
