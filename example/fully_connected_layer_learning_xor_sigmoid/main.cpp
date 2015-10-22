#include <iostream>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/activation_func/sigmoid_loss.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_layer.hpp>
//#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/layer.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	//std::random_device device; std::mt19937 rand(device());
	std::mt19937 rand(0);

	auto input_dim = 2u;
	auto output_dim = 1u;
	auto batch_size = 4u;

	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{0.f}, {1.f}, {1.f}, {0.f}
	};

	neu::gpu_vector input;
	for(auto const& cpui : cpu_input) {
		input.insert(input.end(), cpui.begin(), cpui.end());
	}
	neu::gpu_vector teach;
	for(auto const& cput : cpu_teach) {
		teach.insert(teach.end(), cput.begin(), cput.end());
	}

	auto fc1_param = neu::fully_connected_layer_parameter()
		.input_dim(input_dim).batch_size(batch_size)
		.output_dim(10);
	auto relu1_param = neu::make_activation_layer_parameter(fc1_param);
	auto fc2_param = neu::make_fully_connected_layer_parameter(relu1_param)
		.output_dim(output_dim);
	auto sigmoid_loss_param = neu::make_activation_layer_parameter(fc2_param);

	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 1.f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	constexpr neu::scalar base_lr = 0.1;
	constexpr neu::scalar momentum = 0.9;
	constexpr neu::scalar weight_decay = 0.;

	auto fc1 = neu::make_fully_connected_layer(fc1_param, fc12_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto relu1 = neu::make_activation_layer<neu::rectifier>(relu1_param);
	auto fc2 = neu::make_fully_connected_layer(fc2_param, fc12_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto sigmoid_loss = neu::make_activation_layer<neu::sigmoid_loss>(sigmoid_loss_param);

	auto layers = std::vector<neu::layer>{
		std::ref(fc1),
		relu1,
		std::ref(fc2),
		sigmoid_loss
	};
	std::ofstream error_log("error.txt");
	std::ofstream output_log("output.txt");
	for(auto i = 0u; i < 100u; ++i) {
		neu::layers_forward(layers, input);
		auto output = layers.back().get_next_input();
		neu::print(output_log, output, batch_size);
		auto error = neu::calc_last_layer_delta(output, teach);
		neu::layers_backward(layers, error);
		fc1.update();
		fc2.update();

		neu::gpu_vector squared_error(error.size());
		boost::compute::transform(error.begin(), error.end(),
			error.begin(), squared_error.begin(),
			boost::compute::multiplies<neu::scalar>());
		auto error_sum = boost::compute::accumulate(
			squared_error.begin(), squared_error.end(), 0.f);
		error_log << i << " " << error_sum << std::endl;
	}
	neu::print(teach); std::cout << "\n";
	neu::print(layers.back().get_next_input());
}
