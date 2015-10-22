#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/vector_io.hpp>
#include <neu/load_data_set/load_mnist.hpp>
#include <neu/image.hpp>
int main() {
	std::cout << "hello world" << std::endl;
	auto data = neu::load_mnist("../../../data/mnist/");
	std::cout << data.size() << std::endl;
	std::cout << data.front().size() << std::endl;
	for(auto label = 0u; label < data.size(); ++label) {
		for(auto i = 1u; i < data[label].size()/100; ++i) {
			neu::save_1ch_image_vector_as_monochro_image(data[label][i], 28,
				"d"+std::to_string(label)+std::to_string(i)+".bmp", 1);
		}
	}
}
