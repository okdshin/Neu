#define NEU_CONVOLUTION_LAYER_USE_VECTORIZED_KERNEL
#include <iostream>
#include <boost/timer.hpp>
#include <neu/neural_net.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	auto nn = neu::make_neural_net_from_yaml("cifar10.yaml");
	nn.init_batch();

	std::ofstream cel_log("cross_entropy_loss.txt");
	boost::timer timer;
	for(auto i = 0u; i < 5000u; ++i) {
		auto batch = nn.get_batch();
		const auto input = batch.train_data;
		const auto teach = batch.teach_data;
		auto make_next_batch_future = ds.async_make_next_batch();

		std::cout << "forward..." << std::endl;
		nn.forward(input);
		std::cout << "forward finished " << timer.elapsed() << std::endl;
		auto output = layers.back().get_next_input();
		auto error = neu::calc_last_layer_delta(output, teach);

		std::cout << "backward..." << std::endl;
		timer.restart();
		nn.backward(error);
		std::cout << "backward finished" << timer.elapsed() << std::endl;

		std::cout << "update..." << std::endl;
		timer.restart();
		nn.update();
		std::cout << "update finished" << timer.elapsed() << std::endl;

		std::cout << "calc error..." << std::endl;
		timer.restart();
		auto cel = neu::cross_entropy_loss(output, teach);
		std::cout << "error calculation finished" << timer.elapsed() << std::endl;
		cel_log << i << "\t" << cel << std::endl;
		std::cout << i << "cross entropy loss: " << cel << std::endl;

		make_next_batch_future.wait();
	}
	std::ofstream nnf("cifar10_trained_nn.yaml");
	nn.print(nnf); //TODO
}
