#include <iostream>
#include <functional>
#include <neu/activate_func/rectifier.hpp>
#include <neu/activate_func/sigmoid.hpp>
#include <neu/layer.hpp>
#include <neu/full_connected_layer.hpp>
#include <neu/activate_layer.hpp>
#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>

int main() {
	std::cout << "hello world" << std::endl;

	/*
	auto fc1 = neu::make_full_connected_layer(1,1,1,
		neu::to_gpu_vector(neu::cpu_vector{10.f}),
		neu::to_gpu_vector(neu::cpu_vector{1.f}),
		neu::fixed_learning_rate_gen(0.01));
	auto fc1_layer = static_cast<neu::layer>(std::ref(fc1));
	*/
	auto ac1_layer = static_cast<neu::layer>(neu::make_activate_layer(1, 1, 1,
		neu::sigmoid()));

	ac1_layer.forward(neu::to_gpu_vector(neu::cpu_vector{10.f}));

	//std::cout << fc1_layer.get_next_input()[0] << std::endl;
	std::cout << ac1_layer.get_next_input()[0] << std::endl;
}
