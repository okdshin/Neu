add_executable( test_load_cifar10  main.cpp )
target_link_libraries( test_load_cifar10
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
)
set_target_properties( test_load_cifar10 PROPERTIES OUTPUT_NAME "a.out" )
