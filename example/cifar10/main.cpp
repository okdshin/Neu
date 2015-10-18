#include <iostream>
#include <boost/timer.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
//#include <neu/learning_rate_gen/subtract_delta_weight.hpp>
//#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
//#include <neu/learning_rate_gen/weight_decay.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/activation_func/sigmoid.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/identity.hpp>
#include <neu/activation_func/softmax.hpp>
#include <neu/activation_layer.hpp>
#include <neu/convolution_layer.hpp>
#include <neu/max_pooling_layer.hpp>
#include <neu/average_pooling_layer.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/layer.hpp>
#include <neu/load_data_set/load_cifar10.hpp>
#include <neu/data_set.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	constexpr auto label_num = 10u;
	constexpr auto data_num_per_label = 10u;
	constexpr auto input_dim = 32u*32u*3u;
	constexpr auto batch_size = label_num * data_num_per_label;

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(3); std::cout << "INFO: fixed random engine" << std::endl;

	auto data = neu::load_cifar10("../../../data/cifar-10-batches-bin/");
	for(auto& labeled : data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.); });
		}
	}
	auto ds = neu::make_data_set(label_num, data_num_per_label, input_dim, data, rand);

	auto conv1_param = neu::convolution_layer_parameter()
		.input_width(32).input_channel_num(3).output_channel_num(32)
		.filter_width(5).stride(1).pad(2).batch_size(batch_size);
	auto pool1_param = neu::make_max_pooling_layer_parameter(conv1_param)
		.filter_width(3).stride(2).pad(1);
	auto relu1_param = neu::make_activation_layer_parameter(pool1_param);
	auto conv2_param = neu::make_convolution_layer_parameter(pool1_param)
		.output_channel_num(32).filter_width(5).stride(1).pad(2);
	auto relu2_param = neu::make_activation_layer_parameter(conv2_param);
	auto pool2_param = neu::make_average_pooling_layer_parameter(conv2_param)
		.filter_width(3).stride(2).pad(1);
	auto conv3_param = neu::make_convolution_layer_parameter(pool2_param)
		.output_channel_num(64).filter_width(5).stride(1).pad(2);
	auto relu3_param = neu::make_activation_layer_parameter(conv3_param);
	auto pool3_param = neu::make_average_pooling_layer_parameter(conv3_param)
		.filter_width(3).stride(2).pad(1);
	auto fc1_param = neu::make_fully_connected_layer_parameter(pool3_param)
		.output_dim(64);
	auto fc2_param = neu::make_fully_connected_layer_parameter(fc1_param)
		.output_dim(label_num);
	auto softmax_param = neu::make_activation_layer_parameter(fc2_param);

	std::cout << "fc2 input_dim: " << fc2_param.input_dim() << std::endl;
	std::cout << "fc2 output_dim: " << fc2_param.output_dim() << std::endl;
	std::cout << "fc2 batch_size: " << fc2_param.batch_size() << std::endl;

	auto conv1_g = [&rand, dist=std::normal_distribution<>(0.f, 0.0001f)]
		() mutable { return dist(rand); };
	auto conv23_g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	neu::scalar base_lr = 0.001;
	neu::scalar momentum = 0.9;
	//neu::scalar weight_decay = 0.004;
	neu::scalar weight_decay = 0.0;
	auto conv1 = neu::make_convolution_layer(conv1_param, conv1_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv1_param.weight_dim(), conv1_param.bias_dim()));
	auto pool1 = neu::make_max_pooling_layer(pool1_param);
	auto relu1 = neu::make_activation_layer(relu1_param, neu::rectifier());
	auto conv2 = neu::make_convolution_layer(conv2_param, conv23_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv2_param.weight_dim(), conv2_param.bias_dim()));
	auto relu2 = neu::make_activation_layer(relu2_param, neu::rectifier());
	auto pool2 = neu::make_uniform_average_pooling_layer(pool2_param);
	auto conv3 = neu::make_convolution_layer(conv3_param, conv23_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv3_param.weight_dim(), conv3_param.bias_dim()));
	auto relu3 = neu::make_activation_layer(relu3_param, neu::rectifier());
	auto pool3 = neu::make_uniform_average_pooling_layer(pool3_param);
	auto fc1 = neu::make_fully_connected_layer(fc1_param, fc12_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto fc2 = neu::make_fully_connected_layer(fc2_param, fc12_g, constant_g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	std::cout << "fc2 output dim directly: " << fc2.get_next_input().size() << std::endl;
	auto softmax = neu::make_activation_layer(softmax_param,
		neu::softmax(label_num, batch_size));

	auto layers = std::vector<neu::layer>{
		std::ref(conv1),
		pool1,
		relu1,
		std::ref(conv2),
		relu2,
		pool2,
		std::ref(conv3),
		relu3,
		pool3,
		std::ref(fc1),
		std::ref(fc2),
		softmax
	};
	std::ofstream error_log("error.txt");
	std::ofstream log("log.txt");
	make_next_batch(ds);
	boost::timer timer;
	for(auto i = 0u; i < 5000u; ++i) {
		timer.restart();
		auto batch = ds.get_batch();
		auto input = batch.train_data;
		auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();

		std::cout << "forward..." << std::endl;
		neu::layers_forward(layers, input);
		std::cout << "forward finished " << timer.elapsed() << std::endl;
		timer.restart();

		if(i%10 == 0) {
			log << "conv1_filters_l2norm: " << neu::l2_norm(conv1.get_filters())
				<< " conv1_bias_l2norm: " << neu::l2_norm(conv1.get_bias()) << std::endl;
			log << "conv2_filters_l2norm: " << neu::l2_norm(conv2.get_filters())
				<< " conv2_bias_l2norm: " << neu::l2_norm(conv2.get_bias()) << std::endl;
			log << "conv3_filters_l2norm: " << neu::l2_norm(conv3.get_filters())
				<< " conv3_bias_l2norm: " << neu::l2_norm(conv3.get_bias()) << std::endl;
			log << "fc1_weight_l2norm: " << neu::l2_norm(fc1.get_weight())
				<< " fc1_weight_l2norm: " << neu::l2_norm(fc1.get_bias()) << std::endl;
			log << "fc2_weight_l2norm: " << neu::l2_norm(fc2.get_weight())
				<< " fc2_weight_l2norm: " << neu::l2_norm(fc2.get_bias()) << std::endl;
		}
		if(i%100 == 0) {
			for(auto j = 0u; j < layers.size(); ++j) {
				auto output = layers[j].get_next_input();
				std::ofstream outputf(
					"output l"+std::to_string(j)+" i"+std::to_string(i)+".txt");
				neu::print(outputf, output, output.size()/batch_size);
			}
		}
		auto output = layers.back().get_next_input();
		volatile auto output_sum = boost::compute::accumulate(output.begin(), output.end(),
			0.f, boost::compute::plus<neu::scalar>());
		std::cout << "output sum: " << output_sum << std::endl;
		neu::gpu_vector error(output.size());
		boost::compute::transform(output.begin(), output.end(),
			teach.begin(), error.begin(), boost::compute::minus<neu::scalar>());

		neu::gpu_vector squared_error(error.size());
		boost::compute::transform(error.begin(), error.end(),
			error.begin(), squared_error.begin(),
			boost::compute::multiplies<neu::scalar>());
		auto squared_error_sum = boost::compute::accumulate(
			squared_error.begin(), squared_error.end(), 0.f);
		std::cout << i << ":squared_error_sum: " << squared_error_sum << std::endl;
		error_log << i << "\t" << squared_error_sum << std::endl;

		std::cout << "backward..." << std::endl;
		neu::layers_backward(layers, error);
		std::cout << "backward finished" << timer.elapsed() << std::endl;
		timer.restart();

		std::cout << "update..." << std::endl;
		conv1.update();
		conv2.update();
		conv3.update();
		fc1.update();
		fc2.update();
		std::cout << "update finished" << timer.elapsed() << std::endl;
		timer.restart();

		make_next_batch_future.wait();
	}
}
