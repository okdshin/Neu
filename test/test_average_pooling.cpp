#define BOOST_TEST_MODULE TestAveragePooling
#include <boost/test/unit_test.hpp>

#include <neu/layer/average_pooling.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	neu::layer::geometric_layer_property glp{3, 1, 1, 1, 1, 0};
	auto apool = neu::layer::make_uniform_average_pooling(glp, 1, queue);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f,
			1.f, 0.f, 1.f,
			0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(apool), context);
	neu::layer::forward(apool, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 9);
	CHECK_RANGE_EQUAL(neu::scalar, 9, output,
		(
			0.f, 1.f, 0.f,
			1.f, 0.f, 1.f,
			0.f, 1.f, 0.f
		));
}

BOOST_AUTO_TEST_CASE(forward2) {
	neu::layer::geometric_layer_property glp{3, 2, 1, 1, 2, 1};
	auto apool = neu::layer::make_uniform_average_pooling(glp, 1, queue);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f,
			1.f, 0.f, 1.f,
			0.f, 1.f, 0.f,
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(apool), context);
	neu::layer::forward(apool, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	-1.f, -1.f,
		 	-1.f, -1.f,
		));
}

BOOST_AUTO_TEST_CASE(forward3) {
	neu::layer::geometric_layer_property glp{5, 3, 1, 1, 1, 1};
	auto apool = neu::layer::make_uniform_average_pooling(glp, 1, queue);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(apool), context);
	neu::layer::forward(apool, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 25);
	CHECK_RANGE_EQUAL(neu::scalar, 25, output,
		(
			2.f/9.f, 3.f/9.f, 3.f/9.f, 3.f/9.f, 2.f/9.f,
			3.f/9.f, 4.f/9.f, 5.f/9.f, 4.f/9.f, 3.f/9.f,
			3.f/9.f, 5.f/9.f, 4.f/9.f, 5.f/9.f, 3.f/9.f,
			3.f/9.f, 4.f/9.f, 5.f/9.f, 4.f/9.f, 3.f/9.f,
			2.f/9.f, 3.f/9.f, 3.f/9.f, 3.f/9.f, 2.f/9.f
		));
}

BOOST_AUTO_TEST_CASE(forward4) {
	neu::layer::geometric_layer_property glp{5, 3, 1, 1, 2, 1};
	auto apool = neu::layer::make_uniform_average_pooling(glp, 1, queue);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 1.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(apool), context);
	neu::layer::forward(apool, input, output, queue);
	queue.finish();
	std::cout << output.size() << std::endl;
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	2.f/9.f, 3.f/9.f,
			3.f/9.f, 4.f/9.f
		));
}

BOOST_AUTO_TEST_CASE(forward5) {
	neu::layer::geometric_layer_property glp{6, 3, 1, 1, 2, 1};
	auto apool = neu::layer::make_uniform_average_pooling(glp, 1, queue);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
			1.f, 0.f, 1.f, 0.f, 1.f, 0.f,
			0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
			1.f, 0.f, 1.f, 0.f, 1.f, 0.f,
			0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
			1.f, 0.f, 1.f, 0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(apool), context);
	neu::layer::forward(apool, input, output, queue);
	queue.finish();
	std::cout << output.size() << std::endl;
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	2.f/9.f, 3.f/9.f,
			3.f/9.f, 4.f/9.f
		));
}
BOOST_AUTO_TEST_SUITE_END()
