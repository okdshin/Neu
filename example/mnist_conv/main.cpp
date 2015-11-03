#include <iostream>
#include <boost/timer.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/image.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/softmax_loss.hpp>
#include <neu/activation_layer.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/max_pooling_layer.hpp>
#include <neu/convolution_layer.hpp>
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

	auto conv1_param = neu::convolution_layer_parameter()
		.input_width(28).input_channel_num(1).batch_size(batch_size)
		.filter_width(5).stride(1).pad(2).output_channel_num(20);
	auto pool1_param = neu::make_max_pooling_layer_parameter(conv1_param)
		.filter_width(2).stride(2).pad(1);
	auto conv2_param = neu::make_convolution_layer_parameter(pool1_param)
		.filter_width(5).stride(1).pad(2).output_channel_num(50);
	auto fc1_param = neu::make_fully_connected_layer_parameter(conv2_param)
		.output_dim(500);
	auto relu1_param = neu::make_activation_layer_parameter(fc1_param);
	auto fc2_param = neu::make_fully_connected_layer_parameter(relu1_param)
		.output_dim(label_num);
	auto softmax_loss_param = neu::make_activation_layer_parameter(fc2_param);

	auto g = [&rand, dist=std::uniform_real_distribution<>(-0.01f, 0.01f)]
		() mutable { return dist(rand); };

	neu::scalar base_lr = 0.001;
	neu::scalar momentum = 0.9;
	neu::scalar weight_decay = 0.0;
	auto conv1 = neu::make_convolution_layer(conv1_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv1_param.weight_dim(), conv1_param.bias_dim()));
	auto pool1 = neu::make_max_pooling_layer(pool1_param);
	auto conv2 = neu::make_convolution_layer(conv2_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv2_param.weight_dim(), conv2_param.bias_dim()));
	auto fc1 = neu::make_fully_connected_layer(fc1_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto relu1 = neu::make_activation_layer(relu1_param, neu::rectifier());
	auto fc2 = neu::make_fully_connected_layer(fc2_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto softmax_loss = neu::make_activation_layer(softmax_loss_param,
		neu::softmax_loss(label_num, batch_size));

	auto layers = std::vector<neu::layer>{
		std::ref(conv1),
		pool1,
		std::ref(conv2),
		std::ref(fc1),
		relu1,
		std::ref(fc2),
		softmax_loss
	};
	std::ofstream error_log("error.txt");
	std::ofstream output_log("output.txt");
	make_next_batch(ds);
	boost::timer timer;
	auto buffer_size = neu::max_inoutput_size(layers);
	neu::gpu_vector buffer1(buffer_size);
	neu::gpu_vector buffer2(buffer_size);
	neu::gpu_vector buffer3(buffer_size);
	for(auto i = 0u; i < 10000u; ++i) {
		std::cout << "iteration: " << i << std::endl;
		timer.restart();
		auto batch = ds.get_batch();
		auto input = batch.train_data;
		auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();
		auto output_range = neu::layers_forward(layers,
			neu::to_range(input), buffer1, buffer2);
		//auto cel = neu::cross_entropy_loss(output_range, teach, buffer3);
		//error_log << i << " " << cel << std::endl;
		neu::print(output_log, output_range, label_num);
		auto error_range = output_range;
		neu::calc_last_layer_delta(output_range, teach, error_range);
		auto mse = neu::mean_square_error(error_range);
		error_log << i << " " << mse << std::endl;
		//neu::print(error_log, error_range, label_num);
		auto prev_delta_range = neu::layers_backward(layers,
			error_range, buffer1, buffer2);
		neu::layers_update(layers);
		make_next_batch_future.wait();
		std::cout << "iteration consumed " << timer.elapsed() << "secs" << std::endl;
	}
}
