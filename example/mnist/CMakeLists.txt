add_executable( mnist main.cpp )
target_link_libraries( mnist
	${Boost_LIBRARIES}
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties( mnist PROPERTIES OUTPUT_NAME "a.out" )
