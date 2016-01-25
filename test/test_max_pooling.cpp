#define BOOST_TEST_MODULE TestMaxPooling
#include <boost/test/unit_test.hpp>

#include <neu/layer/max_pooling.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	neu::layer::geometric_layer_property glp{3, 1, 1, 1, 1, 0};
	neu::layer::max_pooling mpool(glp, 1, context);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f,
			1.f, 0.f, 1.f,
			0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(mpool), context);
	neu::layer::forward(mpool, input, output, queue);
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
	neu::layer::max_pooling mpool(glp, 1, context);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f,
			1.f, 0.f, 1.f,
			0.f, 1.f, 0.f,
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(mpool), context);
	neu::layer::forward(mpool, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	0.f, 1.f,
		 	1.f, 1.f,
		));
}

BOOST_AUTO_TEST_CASE(forward3) {
	neu::layer::geometric_layer_property glp{5, 3, 1, 1, 2, 1};
	neu::layer::max_pooling mpool(glp, 1, context);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(mpool), context);
	neu::layer::forward(mpool, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	1.f, 1.f,
			1.f, 1.f
		));
}

BOOST_AUTO_TEST_CASE(forward4) {
	neu::layer::geometric_layer_property glp{5, 5, 1, 1, 2, 2};
	neu::layer::max_pooling mpool(glp, 1, context);
	neu::gpu_vector input(
		{
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f,
			1.f, 0.f, 1.f, 0.f, 1.f,
			0.f, 1.f, 0.f, 1.f, 0.f
		},
		queue);
	neu::gpu_vector output(neu::layer::whole_output_size(mpool), context);
	neu::layer::forward(mpool, input, output, queue);
	queue.finish();
	std::cout << output.size() << std::endl;
	BOOST_CHECK(output.size() == 4);
	CHECK_RANGE_EQUAL(neu::scalar, 4, output,
		(
		 	1.f, 1.f,
			1.f, 1.f
		));
}

BOOST_AUTO_TEST_SUITE_END()
