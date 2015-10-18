#include <iostream>
#include <boost/timer.hpp>
#include <neu/vector_io.hpp>
#include <neu/layers_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/image.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/activation_func/tanh.hpp>
#include <neu/activation_func/sigmoid.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/identity.hpp>
#include <neu/activation_func/softmax.hpp>
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

	std::random_device rd; std::mt19937 rand(rd());
	//std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

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
	auto tanh1_param = neu::make_activation_layer_parameter(fc1_param);
	auto fc2_param = neu::make_fully_connected_layer_parameter(fc1_param)
		.output_dim(label_num);
	auto softmax_param = neu::make_activation_layer_parameter(fc2_param);

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	neu::scalar base_lr = 0.001;
	neu::scalar momentum = 0.0;
	neu::scalar weight_decay = 0.0;
	auto fc1 = neu::make_fully_connected_layer(fc1_param, g, g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim()));
	auto tanh1 = neu::make_activation_layer(tanh1_param, neu::tanh());
	auto fc2 = neu::make_fully_connected_layer(fc2_param, g, g,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc2_param.weight_dim(), fc2_param.bias_dim()));
	auto softmax = neu::make_activation_layer(softmax_param,
		neu::softmax(label_num, batch_size));

	auto layers = std::vector<neu::layer>{
		std::ref(fc1),
		tanh1,
		std::ref(fc2),
		softmax
	};
	std::ofstream error_log("error.txt");
	std::ofstream error_vec_log("error_vec.txt");
	std::ofstream log("log.txt");
	make_next_batch(ds);
	boost::timer timer;
	for(auto i = 0u; i < 100000u; ++i) {
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
			std::ofstream teachf("teach i"+std::to_string(i)+".txt");
			neu::print(teachf, teach, teach.size()/batch_size);
		}
		auto output = layers.back().get_next_input();
		volatile auto output_sum = boost::compute::accumulate(output.begin(), output.end(),
			0.f, boost::compute::plus<neu::scalar>());
		std::cout << "output sum: " << output_sum << std::endl;
		neu::gpu_vector error(output.size());
		boost::compute::transform(output.begin(), output.end(),
			teach.begin(), error.begin(), boost::compute::minus<neu::scalar>());
		neu::print(error_vec_log, error, label_num);

		/*
		neu::gpu_vector loss(output.size());
		boost::compute::transform(output.begin(), output.end(),
			teach.begin(), loss.begin(), cross_entropy_kernel);
		*/


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
		fc1.update();
		fc2.update();
		std::cout << "update finished" << timer.elapsed() << std::endl;
		timer.restart();

		make_next_batch_future.wait();
	}
}
