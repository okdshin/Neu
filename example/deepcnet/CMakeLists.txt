add_executable( deepcnet main.cpp )
target_link_libraries( deepcnet
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${Boost_PROGRAM_OPTIONS_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${YAMLCPP_LIBRARY}
)
set_target_properties( deepcnet PROPERTIES OUTPUT_NAME "a.out" )
