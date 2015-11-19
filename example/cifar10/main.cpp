//#define NEU_DISABLE_ASSERTION
//#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/activation_func/sigmoid.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/softmax_loss.hpp>
#include <neu/activation_layer.hpp>
#include <neu/convolution_layer.hpp>
#include <neu/max_pooling_layer.hpp>
#include <neu/average_pooling_layer.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/layer.hpp>
#include <neu/load_data_set/load_cifar10.hpp>
#include <neu/data_set.hpp>

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	constexpr auto label_num = 10u;
	constexpr auto input_dim = 32u*32u*3u;
	constexpr auto test_data_num_per_label = 1000;

	int data_num_per_label;
	int iteration_limit;
	neu::scalar base_lr;
	neu::scalar momentum;
	//neu::scalar weight_decay = 0.004;
	neu::scalar weight_decay;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("data_num_per_label", po::value<int>(&data_num_per_label)->default_value(10),
		 "set number of data per label for Batch SGD")
		("iteration_limit", po::value<int>(&iteration_limit)->default_value(5000), 
		 "set training iteration limit")
		("base_lr", po::value<neu::scalar>(&base_lr)->default_value(0.001), 
		 "set base learning rate")
		("momentum", po::value<neu::scalar>(&momentum)->default_value(0.9), 
		 "set momentum rate")
		("weight_decay", po::value<neu::scalar>(&weight_decay)->default_value(0.), 
		 "set weight decay rate")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	auto batch_size = label_num * data_num_per_label;
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}
	std::cout << "data_num_per_label was set to " << data_num_per_label << ".";
	std::cout << "(so batch_size was set to 10*" << data_num_per_label 
		<< "=" << batch_size << ".)\n";
	std::cout << "iteration_limit was set to " << iteration_limit << ".\n";
	std::cout << "base_lr was set to " << base_lr << ".\n";
	std::cout << "momentum was set to " << momentum << ".\n";
	std::cout << "weight_decay was set to " << weight_decay << ".\n";


	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(3); std::cout << "INFO: fixed random engine" << std::endl;

	auto train_data = neu::load_cifar10_train_data("../../../data/cifar-10-batches-bin/");
	for(auto& labeled : train_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.); });
		}
	}
	auto train_ds = neu::make_data_set(
		label_num, data_num_per_label, input_dim, train_data, rand);

	auto test_data = neu::load_cifar10_test_data("../../../data/cifar-10-batches-bin/");
	for(auto& labeled : test_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.); });
		}
	}
	auto test_ds = neu::make_data_set(
		label_num, test_data_num_per_label, input_dim, test_data, rand);

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
	auto softmax_loss_param = neu::make_activation_layer_parameter(fc2_param);

	auto conv1_g = [&rand, dist=std::normal_distribution<>(0.f, 0.0001f)]
		() mutable { return dist(rand); };
	auto conv23_g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	auto conv1 = neu::make_convolution_layer(conv1_param, conv1_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv1_param.weight_dim(), conv1_param.bias_dim()));
	auto pool1 = neu::make_max_pooling_layer(pool1_param);
	auto relu1 = neu::make_activation_layer(relu1_param, neu::rectifier());
	auto conv2 = neu::make_convolution_layer(conv2_param, conv23_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv2_param.weight_dim(), conv2_param.bias_dim()));
	auto relu2 = neu::make_activation_layer(relu2_param, neu::rectifier());
	auto pool2 = neu::make_uniform_average_pooling_layer(pool2_param);
	auto conv3 = neu::make_convolution_layer(conv3_param, conv23_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv3_param.weight_dim(), conv3_param.bias_dim()));
	auto relu3 = neu::make_activation_layer(relu3_param, neu::rectifier());
	auto pool3 = neu::make_uniform_average_pooling_layer(pool3_param);
	auto fc1 = neu::make_fully_connected_layer(fc1_param, fc12_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto fc2 = neu::make_fully_connected_layer(fc2_param, fc12_g, constant_g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto softmax_loss = neu::make_activation_layer(softmax_loss_param,
		neu::softmax_loss(label_num, batch_size));

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
		softmax_loss
	};
	/*
	std::ofstream mse_log("mean_square_error.txt");
	std::ofstream cel_log("cross_entropy_loss.txt");
	std::ofstream test_accuracy_log("test_accuracy.txt");
	std::ofstream test_cel_log("test_cross_entropy_loss.txt"); std::ofstream log("log.txt");
	*/
	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream output_log("output.txt");
	make_next_batch(train_ds);
	make_next_batch(test_ds);
	neu::gpu_vector output;
	neu::gpu_vector prev_delta;
	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0; i < iteration_limit; ++i) {
		auto batch = train_ds.get_batch();
		const auto input = batch.train_data;
		const auto teach = batch.teach_data;
		auto make_next_batch_future = train_ds.async_make_next_batch();

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
		neu::layers_update(layers);

		make_next_batch_future.wait();
		++progress;
	}
	std::cout << timer.elapsed() << " secs" << std::endl;
}
