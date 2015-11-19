//#define NEU_DISABLE_ASSERTION
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/image.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/tanh.hpp>
#include <neu/activation_func/sigmoid.hpp>
#include <neu/activation_func/sigmoid_loss.hpp>
#include <neu/activation_func/softmax_loss.hpp>
#include <neu/activation_layer.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/layer.hpp>
#include <neu/load_data_set/load_mnist.hpp>
#include <neu/data_set.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	constexpr auto label_num = 10u;
	constexpr auto data_num_per_label = 10u;
	constexpr auto input_dim = 28u*28u*1u;
	constexpr auto batch_size = label_num * data_num_per_label;

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

	auto data = neu::load_mnist("../../../data/mnist/");
	for(auto& labeled : data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(), [](auto e){ return e/255.f; });
		}
	}
	auto ds = neu::make_data_set(label_num, data_num_per_label, input_dim, data, rand);

	auto fc1_param = neu::fully_connected_layer_parameter()
		.input_dim(input_dim).batch_size(batch_size)
		.output_dim(100);
	auto ac1_param = neu::make_activation_layer_parameter(fc1_param);
	auto fc2_param = neu::make_fully_connected_layer_parameter(ac1_param)
		.output_dim(label_num);
	auto softmax_loss_param = neu::make_activation_layer_parameter(fc2_param);

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	neu::scalar base_lr = 0.001;
	neu::scalar momentum = 0.0;
	neu::scalar weight_decay = 0.0;
	auto fc1 = neu::make_fully_connected_layer(fc1_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto ac1 = neu::make_activation_layer(ac1_param, neu::rectifier());
	//auto ac1 = neu::make_activation_layer(ac1_param, neu::tanh());
	//auto ac1 = neu::make_activation_layer(ac1_param, neu::sigmoid());
	auto fc2 = neu::make_fully_connected_layer(fc2_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto softmax_loss = neu::make_activation_layer(softmax_loss_param,
		neu::softmax_loss(label_num, batch_size));
		//neu::sigmoid_loss());

	auto layers = std::vector<neu::layer>{
		std::ref(fc1),
		ac1,
		std::ref(fc2),
		softmax_loss
	};
	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream cel_error_log2("cel_error2.txt");
	std::ofstream output_log("output.txt");
	std::ofstream del_weight_log("del_weight.txt");
	make_next_batch(ds);
	neu::gpu_vector output;
	neu::gpu_vector prev_delta;
	auto iteration_limit = 10000u;
	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0u; i < iteration_limit; ++i) {
		auto batch = ds.get_batch();
		auto input = batch.train_data;
		auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();

		neu::layers_forward(layers, input, output);
		{
			neu::print(output_log, output, label_num);

			auto mse = neu::mean_square_error(output, teach);
			mse_error_log << i << " " << mse << std::endl;

			auto cel = neu::cross_entropy_loss(output, teach);
			cel_error_log << i << " " << cel << std::endl;
		}
		neu::gpu_vector error(output.size());
		neu::calc_last_layer_delta(output, teach, error);
		neu::layers_backward(layers, error, prev_delta);
		{
			auto cel2 = neu::cross_entropy_loss(prev_delta, input);
			cel_error_log2 << i << " " << cel2 << std::endl;
		}
		neu::layers_update(layers);
		{
			del_weight_log << i << " " << neu::range_sum(fc1.del_weight()) << std::endl;
			del_weight_log << i << " " << neu::range_sum(fc1.del_bias()) << std::endl;
		}

		make_next_batch_future.wait();
		++progress;
	}
	std::cout << timer.elapsed() << " secs" << std::endl;
	boost::compute::system::finish();
}
