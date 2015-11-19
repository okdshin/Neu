# find Boost
find_package(Boost 1.58 COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

subdirs( xor_sigmoid )
subdirs( xor_softmax )
subdirs( cifar10 )
subdirs( mnist )
subdirs( mnist_conv )
subdirs( convolution )
subdirs( convolution_layer )
subdirs( max_pooling_layer )
subdirs( average_pooling_layer )
