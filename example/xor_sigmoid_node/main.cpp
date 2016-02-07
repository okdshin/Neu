//#define NEU_DISABLE_ASSERTION
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <neu/vector_io.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/activation/rectifier.hpp>
#include <neu/layer/activation/sigmoid_loss.hpp>
#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/io.hpp>
#include <neu/node/graph.hpp>
#include <neu/node/fixed_data.hpp>
#include <neu/node/layer.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;
	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	std::random_device device; std::mt19937 rand(device());
	//std::mt19937 rand(0);

	const auto input_dim = 2u;
	const auto output_dim = 1u;
	const auto batch_size = 4u;

	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{0.f}, {1.f}, {1.f}, {0.f}
	};

	const auto input = neu::flatten(cpu_input);
	const auto teach = neu::flatten(cpu_teach);

	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 1.f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };

	constexpr neu::scalar base_lr = 0.1f;
	//constexpr neu::scalar momentum = 0.9f;
	constexpr std::size_t hidden_node_num = 10u;

	auto opt = neu::optimizer::fixed_learning_rate(base_lr);

	neu::node::graph graph;
	graph.add_node("input", neu::node::fixed_data(input, queue));
	graph.add_node("ip1",
		neu::node::layer(
			neu::layer::make_inner_product(
				input_dim, hidden_node_num, batch_size, fc12_g,
				/*neu::optimizer::momentum(
					base_lr, momentum, input_dim*hidden_node_num, queue
				),*/
				opt,
				queue
			),
			queue
		)
	);
	graph.add_node("bias1",
		neu::node::layer(
			neu::layer::make_bias(
				hidden_node_num, batch_size, constant_g,
				/*neu::optimizer::momentum(
					base_lr, momentum, hidden_node_num, queue
				),*/
				opt,
				queue
			),
			queue
		)
	);
	graph.add_node(
		"relu",
		neu::node::layer(
			neu::layer::make_rectifier(hidden_node_num, batch_size),
			queue
		)
	);
	graph.add_node(
		"ip2",
		neu::node::layer(
			neu::layer::make_inner_product(
				hidden_node_num, output_dim, batch_size, fc12_g,
				/*neu::optimizer::momentum(
					base_lr, momentum, hidden_node_num*output_dim, queue),*/
				opt,
				queue
			),
			queue
		)
	);
	graph.add_node("bias2",
		neu::node::layer(
			neu::layer::make_bias(
				output_dim, batch_size, constant_g,
				/*neu::optimizer::momentum(
					base_lr, momentum, output_dim, queue
				),*/
				opt,
				queue
			),
			queue
		)
	);
	graph.add_node(
		"sigmoid_loss",
		neu::node::layer(
			neu::layer::make_sigmoid_loss(output_dim, batch_size),
			queue
		)
	);
	graph.add_node(
		"error",
		neu::node::error_for_fixed_data(teach, queue)
	);
	neu::node::connect_flow(graph,
		{"input", "ip1", "bias1", "relu", "ip2", "bias2", "sigmoid_loss", "error"});
	//queue.finish();

	auto iteration_limit = 100;

	boost::progress_display progress(iteration_limit);
	boost::timer timer;
	for(auto i = 0; i < iteration_limit; ++i) {
		graph["input"].forward(queue);

		graph["ip1"].forward(queue);
		graph["bias1"].forward(queue);
		graph["relu"].forward(queue);
		graph["ip2"].forward(queue);
		graph["bias2"].forward(queue);
		graph["sigmoid_loss"].forward(queue);

		graph["error"].backward(queue);

		graph["sigmoid_loss"].backward(queue);
		graph["bias2"].backward(queue);
		graph["ip2"].backward(queue);
		graph["relu"].backward(queue);
		graph["bias1"].backward(queue);
		graph["ip1"].backward(queue);

		graph["ip1"].update(queue);
		graph["bias1"].update(queue);
		graph["ip2"].update(queue);
		graph["bias2"].update(queue);
		++progress;
	}
	std::cout << timer.elapsed() << " sec" << std::endl;
	boost::compute::system::finish();
	auto output = graph["sigmoid_loss"].output_for(&graph["error"]);
	for(auto e : output) {
		std::cout << e << " ";
	}
	std::cout << std::endl;
}
