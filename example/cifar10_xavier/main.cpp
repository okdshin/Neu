//#define NEU_DISABLE_ASSERTION
//#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
#define NEU_BENCHMARK_ENABLE
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <neu/vector_io.hpp>
#include <neu/kernel.hpp>
#include <neu/image.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/activation/rectifier.hpp>
#include <neu/layer/activation/softmax_loss.hpp>
//#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/average_pooling.hpp>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/io.hpp>
#include <neu/dataset/load_cifar10.hpp>
#include <neu/dataset/classification_dataset.hpp>

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	constexpr auto label_num = 10u;
	constexpr auto input_width = 32u;
	constexpr auto input_channel_num = 3u;
	constexpr auto test_data_num_per_label = 1000;

	constexpr auto input_dim = input_width*input_width*input_channel_num;

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

	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	std::random_device rd; std::mt19937 rand(rd());
	//std::mt19937 rand(1); std::cout << "INFO: fixed random engine" << std::endl;

	auto train_data = neu::dataset::load_cifar10_train_data("../../../data/cifar-10-batches-bin/");
	for(auto& labeled : train_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.)/127.f; });
		}
	}
	auto train_ds = neu::dataset::make_classification_dataset(
		label_num, data_num_per_label, input_dim, train_data, rand, context);

	auto test_data =
		neu::dataset::load_cifar10_test_data("../../../data/cifar-10-batches-bin/");
	for(auto& labeled : test_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.f)/127.f; });
		}
	}
	auto test_ds = neu::dataset::make_classification_dataset(
		label_num, test_data_num_per_label, input_dim, test_data, rand, context);

	auto rng = rand;
	auto constant_g = [](){ return 0.f; };

	std::vector<neu::layer::any_layer> nn;

	{ // conv1
		neu::layer::geometric_layer_property glp{
			input_width,
			5,
			input_channel_num,
			32,
			1, 2
		};
		nn.push_back(neu::layer::make_convolution_xavier(
			glp, batch_size, rng, neu::optimizer::fixed_learning_rate(base_lr), queue));
	}
	auto conv1 = nn.back();

	{ // bias-2
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // pool1(max)
		neu::layer::geometric_layer_property glp{
			neu::layer::output_width(conv1),
			3,
			neu::layer::output_channel_num(conv1),
			neu::layer::output_channel_num(conv1),
			2, 1
		};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
	}
	auto pool1 = nn.back();

	{ // relu1
		nn.push_back(neu::layer::make_rectifier(neu::layer::output_dim(nn), batch_size));
	}

	{ // conv2
		neu::layer::geometric_layer_property glp{
			neu::layer::output_width(pool1),
			5,
			neu::layer::output_channel_num(pool1),
			32,
			1, 2
		};
		nn.push_back(neu::layer::make_convolution_xavier(
			glp, batch_size, rng,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}
	auto conv2 = nn.back();

	{ // bias-1
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // relu2
		nn.push_back(neu::layer::make_rectifier(neu::layer::output_dim(nn), batch_size));
	}

	{ // pool2(ave)
		neu::layer::geometric_layer_property glp{
			neu::layer::output_width(conv2),
			3,
			neu::layer::output_channel_num(conv2),
			neu::layer::output_channel_num(conv2),
			2, 1
		};
		nn.push_back(neu::layer::make_uniform_average_pooling(glp, batch_size, queue));
	}

	{ // conv3
		neu::layer::geometric_layer_property glp{
			neu::layer::output_width(nn),
			5,
			neu::layer::output_channel_num(nn),
			64,
			1, 2
		};
		nn.push_back(neu::layer::make_convolution_xavier(
			glp, batch_size, rng,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}
	auto conv3 = nn.back();

	{ // bias0
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // relu3
		nn.push_back(neu::layer::make_rectifier(neu::layer::output_dim(nn), batch_size));
	}

	{ // pool3(ave)
		neu::layer::geometric_layer_property glp{
			neu::layer::output_width(conv3),
			3,
			neu::layer::output_channel_num(conv3),
			neu::layer::output_channel_num(conv3),
			2, 1
		};
		nn.push_back(neu::layer::make_uniform_average_pooling(glp, batch_size, queue));
	}

	{ // ip1
		nn.push_back(neu::layer::make_inner_product_xavier(
			neu::layer::output_dim(nn), 64, batch_size, rng,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // bias1
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // ip2
		nn.push_back(neu::layer::make_inner_product_xavier(
			neu::layer::output_dim(nn), label_num, batch_size, rng,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // bias2
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // softmax_loss
		nn.push_back(neu::layer::make_softmax_loss(
			neu::layer::output_dim(nn), batch_size, context));
	}

	neu::layer::output_to_file(nn, "nn_before.yaml", queue);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream output_log("output.txt");
	make_next_batch(train_ds);
	make_next_batch(test_ds);

	neu::gpu_vector output(neu::layer::whole_output_size(nn), context);
	neu::gpu_vector error(output.size(), context);
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(nn), context);

	boost::progress_display progress(iteration_limit);
	boost::timer timer;

	for(auto i = 0; i < iteration_limit; ++i) {
		auto batch = train_ds.get_batch();
		const auto input = batch.train_data;
		const auto teach = batch.teach_data;
		auto make_next_batch_future = train_ds.async_make_next_batch();

		if(i%(iteration_limit/10) == 0) {
			neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);
		}

		neu::layer::forward(nn, input, output, queue);

		{
			neu::print(output_log, output, label_num);

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
	std::cout << timer.elapsed() << " secs" << std::endl;
	boost::compute::system::finish();
	neu::layer::output_to_file(nn, "nn.yaml", queue);
}
