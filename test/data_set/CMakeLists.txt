add_executable( test_data_set main.cpp )
target_link_libraries( test_data_set
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( test_data_set PROPERTIES OUTPUT_NAME "a.out" )
