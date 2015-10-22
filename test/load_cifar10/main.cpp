#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/vector_io.hpp>
#include <neu/load_data_set/load_cifar10.hpp>
#include <neu/image.hpp>
int main() {
	std::cout << "hello world" << std::endl;
	auto data = neu::load_cifar10("../../../data/cifar-10-batches-bin/");
	std::cout << data.size() << std::endl;
	std::cout << data.front().size() << std::endl;
	for(auto label = 0u; label < 3u; ++label) {
		for(auto i = 0u; i < data[label].size(); ++i) {
			neu::save_3ch_image_vector_as_rgb_image(data[label][i], 32,
				"d"+std::to_string(label)+std::to_string(i)+".bmp");
		}
	}
}
