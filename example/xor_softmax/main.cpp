//#define NEU_DISABLE_ASSERTION
#include <iostream>
#include <boost/progress.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/activation_func/softmax_loss.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_layer.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/layer.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	//std::random_device device; std::mt19937 rand(device());
	std::mt19937 rand(0);

	auto input_dim = 2u;
	auto output_dim = 2u;
	auto batch_size = 4u;

	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{1.f, 0.f}, {0.f, 1.f}, {0.f, 1.f}, {1.f, 0.f}
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
	auto softmax_loss_param = neu::make_activation_layer_parameter(fc2_param);

	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 1.f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	neu::scalar base_lr = 0.01;
	neu::scalar momentum = 0.9;
	neu::scalar weight_decay = 0.;

	auto fc1 = neu::make_fully_connected_layer(fc1_param, fc12_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto relu1 = neu::make_activation_layer(relu1_param, neu::rectifier());
	auto fc2 = neu::make_fully_connected_layer(fc2_param, fc12_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto softmax_loss = neu::make_activation_layer(softmax_loss_param,
		neu::softmax_loss(output_dim, batch_size));

	auto layers = std::vector<neu::layer>{
		std::ref(fc1),
		relu1,
		std::ref(fc2),
		softmax_loss
	};
	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream output_log("output.txt");

	neu::gpu_vector output;
	neu::gpu_vector prev_delta;
	auto iteration_limit = 500u;
	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0u; i < iteration_limit; ++i) {
		neu::layers_forward(layers, input, output);
		{
			neu::print(output_log, output, output_dim);
			auto queue = boost::compute::system::default_queue();
			auto mse = neu::mean_square_error(output, teach, queue);
			mse_error_log << i << " " << mse << std::endl;
		}
		neu::gpu_vector error(output.size());
		neu::calc_last_layer_delta(output, teach, error);
		neu::layers_backward(layers, error, prev_delta);
		neu::layers_update(layers);
		++progress;
	}
	std::cout << timer.elapsed() << " sec" << std::endl;
	neu::print(std::cout, output, output_dim);
	boost::compute::system::finish();
}
