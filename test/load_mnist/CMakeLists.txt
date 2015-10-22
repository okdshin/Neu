add_executable( test_load_mnist  main.cpp )
target_link_libraries( test_load_mnist
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( test_load_mnist PROPERTIES OUTPUT_NAME "a.out" )
