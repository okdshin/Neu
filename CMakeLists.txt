cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(neu)
enable_testing() 
include(CheckIncludeFiles)
include (FindPkgConfig)
include(CheckCXXSourceCompiles)

# build type
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -Og -g -DGSL_THROWS_FOR_TESTING -DBOOST_COMPUTE_THREAD_SAFE")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -Og -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -O2 -DBOOST_COMPUTE_THREAD_SAFE")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -Wno-unused-parameter -Werror -O2")

find_package( Boost COMPONENTS system thread timer filesystem program_options REQUIRED)
if( NOT Boost_SYSTEM_FOUND )
	message( SEND_ERROR "Required package Boost.System is not detected." )
endif (NOT Boost_SYSTEM_FOUND)
if( NOT Boost_THREAD_FOUND )
	message( SEND_ERROR "Required package Boost.Thread is not detected." )
endif (NOT Boost_THREAD_FOUND)
if( NOT Boost_TIMER_FOUND )
	message( SEND_ERROR "Required package Boost.Timer is not detected." )
endif (NOT Boost_TIMER_FOUND)
if( NOT Boost_FILESYSTEM_FOUND )
	message( SEND_ERROR "Required package Boost.Filesystem is not detected." )
endif (NOT Boost_FILESYSTEM_FOUND)
if( NOT Boost_PROGRAM_OPTIONS_FOUND )
	message( SEND_ERROR "Required package Boost.ProgramOptions is not detected." )
endif (NOT Boost_PROGRAM_OPTIONS_FOUND)

find_package( BoostCompute REQUIRED )
if( NOT BoostCompute_FOUND )
	message( SEND_ERROR "Required package Boost.Compute is not detected." )
endif (NOT BoostCompute_FOUND)

find_package( OpenCL REQUIRED )
if( NOT OpenCL_FOUND )
	message( SEND_ERROR "Required package OpenCL is not detected." )
endif (NOT OpenCL_FOUND)

find_package( freeimageplus REQUIRED )
if( NOT FREEIMAGEPLUS_FOUND )
	message( SEND_ERROR "Required package FreeImagePlus is not detected." )
endif (NOT FREEIMAGEPLUS_FOUND)

find_package( Threads REQUIRED )
if( NOT Threads_FOUND )
	message( SEND_ERROR "Required package Threads is not detected." )
endif (NOT Threads_FOUND)

find_package(YamlCpp REQUIRED )
if( NOT YAMLCPP_FOUND )
	message( SEND_ERROR "Required package YamlCpp is not detected." )
endif (NOT YAMLCPP_FOUND)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${BoostCompute_INCLUDE_DIRS} ${FREEIMAGEPLUS_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIR})
link_directories( ${Boost_LIBRARY_DIRS} ${OpenCL_LIBRARY_DIRS} ${FREEIMAGEPLUS_LIBRARY_RELEASE} )

install( DIRECTORY ./neu/ DESTINATION ./neu/ FILES_MATCHING PATTERN "*.hpp" )

subdirs( test example )
