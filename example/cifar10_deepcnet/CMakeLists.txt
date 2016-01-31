add_executable( cifer10_deepcnet main.cpp )
target_link_libraries( cifer10_deepcnet
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${Boost_PROGRAM_OPTIONS_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${YAMLCPP_LIBRARY}
)
set_target_properties( cifer10_deepcnet PROPERTIES OUTPUT_NAME "a.out" )
