//#define NEU_DISABLE_ASSERTION
#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
#define NEU_LAYER_SERIALIZE_WITHOUT_LONG_VECTOR
//#define NEU_BENCHMARK_ENABLE
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

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("data_num_per_label", po::value<int>(&data_num_per_label)->default_value(6),
		 "set number of data per label for Batch SGD")
		("iteration_limit", po::value<int>(&iteration_limit)->default_value(500000), 
		 "set training iteration limit")
		("base_lr", po::value<neu::scalar>(&base_lr)->default_value(1.0), 
		 "set base learning rate")
		("momentum", po::value<neu::scalar>(&momentum)->default_value(0.9), 
		 "set momentum rate")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	const auto batch_size = label_num * data_num_per_label;
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

	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	//std::random_device rd; std::mt19937 rand(rd());
	std::mt19937 rand(0); std::cout << "INFO: fixed random engine" << std::endl;

	auto optgen = [base_lr, momentum, &queue](int theta_size) {	
		return neu::optimizer::momentum(
			base_lr, momentum, theta_size, queue);
	};
	std::vector<neu::layer::any_layer> nn;
	neu::layer::ready_made::make_deepcnet(
		nn, batch_size, input_width, label_num, 5, 300, rand, optgen, queue);

	neu::layer::output_to_file(nn, "nn0.yaml", queue);

}
