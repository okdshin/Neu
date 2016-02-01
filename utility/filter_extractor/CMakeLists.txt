add_executable( filter_extractor main.cpp )
include_directories(${YAMLCPP_INCLUDE_DIR})
target_link_libraries( filter_extractor
	${Boost_LIBRARIES}
	${Boost_SYSTEM_LIBRARIES}
	${Boost_FILESYSTEM_LIBRARIES}
	${OpenCL_LIBRARIES}
	${FREEIMAGEPLUS_LIBRARIES}
	${YAMLCPP_LIBRARY}
)
set_target_properties( filter_extractor PROPERTIES OUTPUT_NAME "a.out" )
