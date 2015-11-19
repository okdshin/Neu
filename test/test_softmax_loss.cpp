#define BOOST_TEST_MODULE TestSoftmaxLoss
#include <boost/test/unit_test.hpp>

#include <neu/activation_func/softmax_loss.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(apply) {
	float data[] = {
		1.f, 0.f, 0.f,
		2.f, 1.f, 1.f,
		10.f, 0.f, 0.f,
	};
	neu::gpu_vector input(data, data+9);
	neu::gpu_vector output(9);
	neu::softmax_loss sl(3, 3);
	sl(input, output);
	BOOST_CHECK_CLOSE(float(output[0]), 0.576117f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[1]), 0.211942f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[2]), 0.211942f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[3]), 0.576117f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[4]), 0.211942f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[5]), 0.211942f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[6]), 0.999909f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[7]), 4.53958e-05f, 1e-3f);
	BOOST_CHECK_CLOSE(float(output[8]), 4.53958e-05f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(apply_derivative) {
	float data[] = {
		1.f, 0.f, 0.f,
		2.f, 1.f, 1.f,
		10.f, 0.f, 0.f,
	};
	neu::gpu_vector input(data, data+9);
	neu::gpu_vector output(9);
	neu::derivative<neu::softmax_loss> sld;
	sld(input, output);
	CHECK_RANGE_EQUAL(float, 9, output, (1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f));
}

BOOST_AUTO_TEST_SUITE_END()
