//#define NEU_DISABLE_ASSERTION
#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
//#define NEU_BENCHMARK_ENABLE
//#define NEU_LAYER_SERIALIZE_WITHOUT_LONG_VECTOR
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
	int layer_num;
	int filter_num;
	bool dropout_on = false;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("data_num_per_label", po::value<int>(&data_num_per_label)->default_value(10),
		 "set number of data per label for Batch SGD")
		("iteration_limit", po::value<int>(&iteration_limit)->default_value(500000), 
		 "set training iteration limit")
		("base_lr", po::value<neu::scalar>(&base_lr)->default_value(0.001), 
		 "set base learning rate")
		("momentum", po::value<neu::scalar>(&momentum)->default_value(0.9), 
		 "set momentum rate")
		("weight_decay", po::value<neu::scalar>(&weight_decay)->default_value(0.), 
		 "set weight decay rate")
		("layer_num", po::value<int>(&layer_num)->default_value(5), 
		 "deepcnet layer num")
		("filter_num", po::value<int>(&filter_num)->default_value(300), 
		 "deepcnet filter num")
		("dropout_on", po::value<bool>(&dropout_on)->default_value(false), 
		 "dropout on")
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
	std::cout << "layer_num was set to " << layer_num << ".\n";
	std::cout << "filter_num was set to " << filter_num << ".\n";
	std::cout << "dropout on was set to " << dropout_on << ".\n";
	{
		std::ofstream logf("setting_log.txt");
		logf << "data_num_per_label was set to " << data_num_per_label << ".";
		logf << "(so batch_size was set to 10*" << data_num_per_label 
			<< "=" << batch_size << ".)\n";
		logf << "iteration_limit was set to " << iteration_limit << ".\n";
		logf << "base_lr was set to " << base_lr << ".\n";
		logf << "momentum was set to " << momentum << ".\n";
		logf << "weight_decay was set to " << weight_decay << ".\n";
		logf << "layer_num was set to " << layer_num << ".\n";
		logf << "filter_num was set to " << filter_num << ".\n";
		logf << "dropout on was set to " << dropout_on << ".\n";
	}

	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

	auto train_data = neu::dataset::load_cifar10_train_data(
		"../../../data/cifar-10-batches-bin/");
	for(auto& labeled : train_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.)/*/255.f*/; });
		}
	}
	auto train_ds = neu::dataset::make_classification_dataset(
		label_num, data_num_per_label, input_dim, train_data, rand, context);

	auto test_data = neu::dataset::load_cifar10_test_data(
		"../../../data/cifar-10-batches-bin/");
	for(auto& labeled : test_data) {
		for(auto& d : labeled) {
			std::transform(d.begin(), d.end(), d.begin(),
				[](auto e){ return (e-127.)/*/255.f*/; });
		}
	}
	auto test_ds = neu::dataset::make_classification_dataset(
		label_num, data_num_per_label, input_dim, test_data, rand, context);

	auto constant_g = [](){ return 0.f; };
	auto optgen = [base_lr, momentum, weight_decay, &queue](int theta_size) {	
		return neu::optimizer::momentum(
			base_lr, momentum, weight_decay, theta_size, queue);
	};
	std::vector<neu::layer::any_layer> nn;
	neu::layer::ready_made::make_deepcnet(nn, batch_size, input_width,
		label_num, layer_num, filter_num, dropout_on ? 0.1f : 0.f, rand, optgen, queue);
	neu::layer::output_to_file(nn, "nn0.yaml", queue);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream cel_error_log("cel_error.txt");
	std::ofstream test_cel_error_log("test_cel_error.txt");
	std::ofstream output_log("output.txt");
	make_next_batch(train_ds);

	neu::gpu_vector output(neu::layer::whole_output_size(nn), context);
	neu::gpu_vector error(output.size(), context);
	//neu::gpu_vector prev_delta(neu::layer::whole_input_size(nn), context);

	boost::progress_display progress(iteration_limit);
	boost::timer timer;

	for(auto i = 0; i < iteration_limit; ++i) {
		auto batch = train_ds.get_batch();
		const auto input = batch.train_data;
		const auto teach = batch.teach_data;
		auto make_next_batch_future = train_ds.async_make_next_batch();
		make_next_batch_future.wait();

		if(i%100 == 0) {
			//neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);

			test_ds.async_make_next_batch().wait();
			auto test_batch = test_ds.get_batch();
			neu::gpu_vector test_output(neu::layer::whole_output_size(nn), context);
			const auto test_input = test_batch.train_data;
			const auto test_teach = test_batch.teach_data;
			neu::layer::test_forward(nn, batch_size, test_input, test_output, queue);
			const auto test_cel =
				neu::range::cross_entropy_loss(test_output, test_teach, queue);
			test_cel_error_log << i << " " << test_cel/batch_size << std::endl;
		}
		//neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);

		neu::layer::forward(nn, input, output, queue);

		{
			if(std::isnan(output[0])) {
				neu::layer::output_to_file(nn, "nn"+std::to_string(i)+".yaml", queue);
			}
			neu::print(output_log, output, label_num);

			const auto cel = neu::range::cross_entropy_loss(output, teach, queue);
			cel_error_log << i << " " << cel/batch_size << std::endl;
		}

		neu::range::calc_last_layer_delta(output, teach, error, queue);
		neu::layer::backward_top(nn, error, queue);
		//neu::layer::backward(nn, error, prev_delta, queue);
		neu::layer::update(nn, queue);

		++progress;
	}
	std::cout << timer.elapsed() << " secs" << std::endl;
	boost::compute::system::finish();
	neu::layer::output_to_file(nn, "nn.yaml", queue);
}
