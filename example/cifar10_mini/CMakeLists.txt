add_executable( cifer10_mini main.cpp )
target_link_libraries( cifer10_mini
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( cifer10_mini PROPERTIES OUTPUT_NAME "a.out" )
