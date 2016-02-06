//#define NEU_DISABLE_ASSERTION
//#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
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
#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/average_pooling.hpp>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/io.hpp>
#include <neu/layer/ready_made/deepcnet.hpp>
#include <neu/dataset/load_cifar10.hpp>
#include <neu/dataset/classification_dataset.hpp>

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	constexpr auto label_num = 10u;
	constexpr auto input_width = 32u;
	constexpr auto input_channel_num = 3u;

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
		("base_lr", po::value<neu::scalar>(&base_lr)->default_value(0.0001), 
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

	auto g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto optgen = [base_lr, momentum, &queue](int weight_dim) {
		return neu::optimizer::fixed_learning_rate(base_lr);
	};

	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 0.01f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	std::vector<neu::layer::any_layer> nn;

	/*
	neu::layer::ready_made::make_deepcnet(
		nn, batch_size, input_width, 1, 10, g, optgen, queue);
	*/
	/*
	{
		const int k = 10;
		{
			// conv
			neu::layer::geometric_layer_property glp{input_width, 3, 3, k, 1, 1};
			nn.push_back(make_convolution(
				glp, batch_size, g, optgen(neu::layer::filters_size(glp)), queue));
		}
		auto output_width = neu::layer::output_width(nn.back());
		auto output_channel_num = neu::layer::output_channel_num(nn.back());

		// bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));

		// leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));

		{
			// max pooling
			neu::layer::geometric_layer_property glp{
				output_width,
				2,
				output_channel_num,
				output_channel_num,
				2, 1
			};
			nn.push_back(neu::layer::max_pooling(glp, batch_size, queue.get_context()));
		}
	}
	*/
	const int output_channel_num = 100;
	int conv_output_width;
	{ // conv
		neu::layer::geometric_layer_property glp{
			input_width, 3, input_channel_num, output_channel_num, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	{ // mpool
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num, output_channel_num, 2, 1};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // conv
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num, output_channel_num*2, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	{ // mpool
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*2, output_channel_num*2, 2, 1};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // conv
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*2, output_channel_num*3, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	{ // mpool
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*3, output_channel_num*3, 2, 1};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // conv
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*3, output_channel_num*4, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	{ // mpool
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*4, output_channel_num*4, 2, 1};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // conv
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*4, output_channel_num*5, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	{ // mpool
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*5, output_channel_num*5, 2, 1};
		nn.push_back(neu::layer::max_pooling(glp, batch_size, context));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // conv
		neu::layer::geometric_layer_property glp{
			conv_output_width, 2, output_channel_num*5, output_channel_num*6, 1, 1};
		nn.push_back(make_convolution(
			glp, batch_size, g, neu::optimizer::fixed_learning_rate(base_lr), queue));
		conv_output_width = neu::layer::output_width(glp);
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			output_dim(nn), batch_size, g, optgen(output_dim(nn)), queue));
	}

	{ // leaky relu
		nn.push_back(neu::layer::make_leaky_rectifier(
			neu::layer::output_dim(nn), batch_size, 0.33f));
	}

	std::cout << "conv_output_width: " << conv_output_width << std::endl;

	{ // ip
		nn.push_back(neu::layer::make_inner_product(
			neu::layer::output_dim(nn), 10, batch_size, fc12_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // bias
		nn.push_back(neu::layer::make_bias(
			neu::layer::output_dim(nn), batch_size, constant_g,
			neu::optimizer::fixed_learning_rate(base_lr),
			queue));
	}

	{ // softmax_loss
		nn.push_back(neu::layer::make_softmax_loss(
			neu::layer::output_dim(nn), batch_size, context));
	}
	neu::layer::output_to_file(nn, "nn.yaml", queue);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream output_log("output.txt");
	make_next_batch(train_ds);

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

		if(i%(iteration_limit/100) == 0) {
			neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);
		}
		//neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);

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
