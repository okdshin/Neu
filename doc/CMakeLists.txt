set(HEADERS
	neu/*.hpp
	neu/optimizer/*.hpp
	neu/layer/geometric_layer_property.hpp
	neu/layer/any_layer.hpp
	neu/layer/convolution/impl.hpp
	neu/layer/max_pooling/impl.hpp
	neu/layer/inner_product/impl.hpp
)

#include_directories(${PROJECT_SOURCE_DIR})
#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${BoostCompute_INCLUDE_DIRS}/compute)
#add_library(transport ${SOURCES})

set(CLDOC "/usr/bin/cldoc" CACHE STRING "User specified '/path/to/cldoc'")
if(CLDOC STREQUAL "")
	find_program(CLDOC cldoc)
elseif(NOT EXISTS ${CLDOC})
	message(FATAL_ERROR "CLDOC was set to '${CLDOC}' but does not exist")
endif()

if(NOT DOCS_OUTPUT_DIR)
	set(DOCS_OUTPUT_DIR "gendocs" CACHE STRING "Generated documentation output directory" FORCE)
endif()

if(NOT DOCS_DIR)
	set(DOCS_DIR "" CACHE STRING "Location of external documentation files to be merged in" FORCE)
endif()

if(NOT ${CLDOC} STREQUAL "CLDOC-NOTFOUND")
  if(NOT ${DOCS_DIR} STREQUAL "")
  	set(CLDOC_MERGE "--merge ${DOCS_DIR}")
  else()
  	set(CLDOC_MERGE "")
  endif()

  add_custom_target(docs COMMAND
    ${CLDOC} generate
	${CMAKE_CXX_FLAGS_DOC}
	-v
	-I${PROJECT_SOURCE_DIR}
	-I${BoostCompute_INCLUDE_DIRS}
    --
    ${CLDOC_MERGE}
    --output ${DOCS_OUTPUT_DIR}
    ${HEADERS}
    ${SOURCES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )

  set_target_properties(docs PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1
  )

  add_custom_target(serve_docs COMMAND
    ${CLDOC} serve -- ${DOCS_OUTPUT_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )

  set_target_properties(docs PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1
  )

  add_dependencies(serve_docs docs)
else()
  message(STATUS
    "Documentation will not be generated"
  )
endif()
