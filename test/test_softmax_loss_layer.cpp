#define BOOST_TEST_MODULE TestSoftmaxLossLayer
#include <boost/test/unit_test.hpp>

#include <neu/basic_type.hpp>
#include <neu/activation_func/softmax_loss.hpp>
#include <neu/activation_layer.hpp>

#include <neu/vector_io.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	auto input_dim = 2u;
	auto output_dim = input_dim;
	auto batch_size = 4u;

	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};

	neu::gpu_vector input;
	for(auto const& cpui : cpu_input) {
		input.insert(input.end(), cpui.begin(), cpui.end());
	}
	neu::gpu_vector teach;
	for(auto const& cput : cpu_teach) {
		teach.insert(teach.end(), cput.begin(), cput.end());
	}

	auto ac_param = neu::activation_layer_parameter()
		.input_dim(input_dim).batch_size(batch_size)
		.output_dim(output_dim);

	auto ac = neu::make_activation_layer(ac_param,
		neu::softmax_loss(input_dim, batch_size));

	neu::gpu_vector output(output_dim*batch_size, context);
	ac.test_forward(batch_size, input, output, queue);
	neu::print(output);
	ac.backward(output, input, queue);
	neu::print(input);
	//TODO
}

BOOST_AUTO_TEST_SUITE_END()
