cmake_minimum_required(VERSION 2.8)

project(Neu)

set(CMAKE_MODULE_PATH ${Neu_SOURCE_DIR}/cmake)

# build type
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -O2 -DNEU_DISABLE_ASSERTION")

set(CMAKE_CXX_FLAGS_DOC "-std=c++1y")

# find Threads
find_package(Threads REQUIRED)
include_directories(SYSTEM ${Threads_INCLUDE_DIRS})

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS})
link_directories(${OPENCL_LIBRARY_DIRS})

# find Boost
find_package(Boost 1.58 COMPONENTS system thread filesystem REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# find Boost.Compute
find_package(BoostCompute REQUIRED)
include_directories(SYSTEM ${BoostCompute_INCLUDE_DIRS})
link_directories(${BoostCompute_LIBRARY_DIRS})
add_definitions(-DBOOST_COMPUTE_THREAD_SAFE)

# find FreeImagePlus
find_package(freeimageplus REQUIRED)
include_directories(SYSTEM ${FREEIMAGEPLUS_INCLUDE_DIRS})
link_directories(${FREEIMAGEPLUS_LIBRARY_RELEASE})

# find YamlCpp
find_package(YamlCpp REQUIRED)
include_directories(SYSTEM ${YAMLCPP_INCLUDE_DIRS})
link_directories(${YAMLCPP_LIBRARY_DIRS})

include_directories(${Neu_SOURCE_DIR})

option(NEU_BUILD_TESTS "Build the Neu tests" ON)
if(${NEU_BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()

option(NEU_BUILD_EXAMPLES "Build the Neu examples" ON)
if(${NEU_BUILD_EXAMPLES})
  add_subdirectory(example)
endif()

option(NEU_BUILD_UTILITIES "Build the Neu utilities" ON)
if(${NEU_BUILD_UTILITIES})
  add_subdirectory(utility)
endif()

add_subdirectory(doc)

# install header files
install( DIRECTORY neu DESTINATION neu FILES_MATCHING PATTERN "*.hpp" )
