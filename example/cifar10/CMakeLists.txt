add_executable( cifer10 main.cpp )
target_link_libraries( cifer10
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( cifer10 PROPERTIES OUTPUT_NAME "a.out" )
