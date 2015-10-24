add_executable( yaml_io main.cpp )
target_link_libraries( yaml_io
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( yaml_io PROPERTIES OUTPUT_NAME "a.out" )
