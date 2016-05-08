# find Boost
find_package(Boost 1.58 COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

#subdirs( convolution )
subdirs(convolution_layer)
subdirs(max_pooling_layer)
subdirs(average_pooling_layer)
subdirs(local_contrast_normalization_layer)
subdirs(dropout_layer)
subdirs(shared_dropout_layer)
#subdirs(deepcnet)

subdirs(xor_sigmoid)
subdirs(xor_softmax)
subdirs(mnist)
subdirs(mnist_conv)
subdirs(mnist_conv_xavier)
subdirs(cifar10)
subdirs(cifar10_xavier)
subdirs(cifar10_deepcnet)

subdirs(xor_sigmoid_node)
