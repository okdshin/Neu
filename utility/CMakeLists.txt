# find Boost
find_package(Boost 1.58 COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

subdirs( filter_extractor )
