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
	constexpr auto input_dim = 28u*28u*1u;
	constexpr auto batch_size = label_num * data_num_per_label;

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

	auto data = neu::dataset::load_mnist("../../../data/mnist/");
	for(auto& labeled : data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return e/255.f; });
		}
	}
	auto ds = neu::dataset::make_classification_dataset(
		label_num, data_num_per_label, input_dim, data, rand, context);

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	constexpr neu::scalar base_lr = 0.001f;
	constexpr neu::scalar momentum = 0.9f;
	constexpr std::size_t hidden_node_num = 100u;

	std::vector<neu::layer::any_layer> nn;
	nn.push_back(neu::layer::make_inner_product(
		input_dim, hidden_node_num, batch_size, g,
		neu::optimizer::momentum(base_lr, momentum, input_dim*hidden_node_num, queue),
		queue));
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, g,
		neu::optimizer::momentum(base_lr, momentum, neu::layer::output_dim(nn), queue),
		queue));
	nn.push_back(neu::layer::make_rectifier(neu::layer::output_dim(nn), batch_size));
	nn.push_back(neu::layer::make_inner_product(
		neu::layer::output_dim(nn), label_num, batch_size, g,
		neu::optimizer::momentum(base_lr, momentum,
			neu::layer::output_dim(nn)*label_num, queue), queue));
	nn.push_back(neu::layer::make_bias(neu::layer::output_dim(nn), batch_size, g,
		neu::optimizer::momentum(base_lr, momentum, neu::layer::output_dim(nn), queue),
		queue));
	nn.push_back(
		neu::layer::make_softmax_loss(neu::layer::output_dim(nn), batch_size, context));

	neu::gpu_vector output(neu::layer::whole_output_size(nn), context);
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(nn), context);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream output_log("output.txt");

	neu::dataset::make_next_batch(ds);

	auto iteration_limit = 10000u;
	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0u; i < iteration_limit; ++i) {
		auto batch = ds.get_batch();
		auto input = batch.train_data;
		auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();

		neu::layer::forward(nn, input, output, queue);
		{
			neu::print(output_log, output, label_num);

			auto mse = neu::range::mean_square_error(output, teach, queue);
			mse_error_log << i << " " << mse << std::endl;

			auto cel = neu::range::cross_entropy_loss(output, teach, queue);
			cel_error_log << i << " " << cel << std::endl;
		}
		neu::gpu_vector error(output.size(), context);
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
