#add_neu_example( "mnist_conv" main.cpp )
add_executable( mnist_conv_xavier main.cpp )
target_link_libraries( mnist_conv_xavier
	${Boost_LIBRARIES}
	${Boost_SYSTEM_LIBRARIES}
	${Boost_THREAD_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${YAMLCPP_LIBRARY}
)
set_target_properties( mnist_conv_xavier PROPERTIES OUTPUT_NAME "a.out" )
