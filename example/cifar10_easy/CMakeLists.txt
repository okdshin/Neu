add_executable( cifer10_easy main.cpp )
target_link_libraries( cifer10_easy
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${YAMLCPP_LIBRARIES}
)
set_target_properties( cifer10_easy PROPERTIES OUTPUT_NAME "a.out" )
