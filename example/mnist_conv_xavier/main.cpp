//#define NEU_DISABLE_ASSERTION
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <neu/vector_io.hpp>
#include <neu/kernel.hpp>
#include <neu/image.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/activation/rectifier.hpp>
#include <neu/layer/activation/softmax_loss.hpp>
#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/io.hpp>
#include <neu/dataset/load_mnist.hpp>
#include <neu/dataset/classification_dataset.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;
	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	constexpr auto label_num = 10u;
	constexpr auto data_num_per_label = 10u;
	constexpr auto input_width = 28u;
	constexpr auto input_channel_num = 1u;

	constexpr auto input_dim = input_width*input_width*input_channel_num;
	constexpr auto batch_size = label_num * data_num_per_label;

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

	auto data = neu::dataset::load_mnist("../../../data/mnist/");
	for(auto& labeled : data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(), [](auto e){ return e/255.f; });
		}
	}
	auto ds = neu::dataset::make_classification_dataset(label_num, data_num_per_label, input_dim, data, rand, context);

	/*
	auto g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	*/
	auto g = rand;
	auto constant_g = [](){ return 0; };

	constexpr neu::scalar base_lr = 0.001;
	constexpr neu::scalar momentum = 0.9;
	constexpr std::size_t hidden_node_num = 50u;

	std::vector<neu::layer::any_layer> nn;

	// conv1
	neu::layer::geometric_layer_property glp1{
		input_width, 5, input_channel_num, 20, 1, 2};
	auto conv1 = neu::layer::make_convolution_xavier(
		glp1, batch_size, g,
		/*
		neu::optimizer::momentum(base_lr, momentum,
			neu::layer::filters_size(glp1), queue),
		*/
		neu::optimizer::fixed_learning_rate(base_lr),
		queue);
	nn.push_back(conv1);
	/*
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, g,
		neu::optimizer::momentum(base_lr, momentum, neu::layer::output_dim(nn), queue),
		queue));
	*/

	// max_pooling
	neu::layer::geometric_layer_property glp15{
		neu::layer::output_width(conv1), 2,
		neu::layer::output_channel_num(conv1),
		neu::layer::output_channel_num(conv1),
		2, 1};
	nn.push_back(neu::layer::max_pooling(glp15, batch_size, context));
	
	//conv2
	neu::layer::geometric_layer_property glp2{
		neu::layer::output_width(nn), 5, neu::layer::output_channel_num(nn), 50, 1, 2};
	nn.push_back(neu::layer::make_convolution_xavier(
		glp2, batch_size, g,
		neu::optimizer::fixed_learning_rate(base_lr),
		queue));

	/*
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, g,
		neu::optimizer::momentum(base_lr, momentum, neu::layer::output_dim(nn), queue),
		queue));
	*/

	// fc1
	nn.push_back(neu::layer::make_inner_product_xavier(
		neu::layer::output_dim(nn), hidden_node_num, batch_size, g,
		neu::optimizer::fixed_learning_rate(base_lr),
		queue));
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, constant_g,
		neu::optimizer::fixed_learning_rate(base_lr),
		queue));
	nn.push_back(neu::layer::make_rectifier(neu::layer::output_dim(nn), batch_size));

	// fc2
	nn.push_back(neu::layer::make_inner_product_xavier(
		neu::layer::output_dim(nn), label_num, batch_size, g,
		neu::optimizer::fixed_learning_rate(base_lr),
		queue));
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, constant_g,
		neu::optimizer::fixed_learning_rate(base_lr),
		queue));
	nn.push_back(neu::layer::make_softmax_loss(
		neu::layer::output_dim(nn), batch_size, context));

	neu::layer::output_to_file(nn, "nn_before.yaml", queue);

	neu::gpu_vector output(neu::layer::whole_output_size(nn), context);
	neu::gpu_vector error(output.size(), context);
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(nn), context);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream output_log("output.txt");

	make_next_batch(ds);

	auto iteration_limit = 10000u;
	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0u; i < iteration_limit; ++i) {
		auto batch = ds.get_batch();
		auto input = batch.train_data;
		auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();

		if(i%(iteration_limit/10) == 0) {
			neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);
		}

		neu::layer::forward(nn, input, output, queue);
		{
			neu::print(output_log << i, output, label_num);

			auto mse = neu::range::mean_square_error(output, teach, queue);
			mse_error_log << i << " " << mse/batch_size << std::endl;

			auto cel = neu::range::cross_entropy_loss(output, teach, queue);
			cel_error_log << i << " " << cel/batch_size << std::endl;
		}
		neu::range::calc_last_layer_delta(output, teach, error, queue);
		neu::layer::backward(nn, error, prev_delta, queue);
		neu::layer::update(nn, queue);

		make_next_batch_future.wait();
		++progress;
	}
	queue.finish();
	std::cout << timer.elapsed() << " secs" << std::endl;
	neu::layer::output_to_file(nn, "nn.yaml", queue);
	//auto loaded_nn = neu::layer::input_from_file("nn.yaml", queue);
	queue.finish();
}
