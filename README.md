Neu
===

Neu is an auto-differention library to construct, train and test deep neural networks.
Neu is written in C++14.

This library is released under the MIT License, see LICENSE.

Dependency
---

- [OpenCL](https://www.khronos.org/opencl/)
- [Boost](http://www.boost.org/)
- [Boost.Compute](https://github.com/boostorg/compute)
- [freeimageplus](http://freeimage.sourceforge.net/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [cldoc(optional)](https://jessevdk.github.io/cldoc/)

Features
---

**Layer**
- inner production
- spacial convolution
- max pooling
- average pooling
- dropout

**Activation**
- ReLU
- leaky ReLU
- sigmoid
- tanh

**Optimizer**
- fixed learning rate
- momentum

Installation
---

```
mkdir build && cd build
cmake ..
make
```
