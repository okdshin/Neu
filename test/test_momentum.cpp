#define BOOST_TEST_MODULE TestMomentum
#include <boost/test/unit_test.hpp>

#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	const auto base_lr = 0.01f;
	const auto mom_rate = 0.9f;
	const auto weight_dim = 3;
	neu::optimizer::momentum mom(base_lr, mom_rate, weight_dim, queue);
	neu::gpu_vector weight({
		1.f, 1.f, 1.f,
	}, queue);
	neu::gpu_vector del_weight({
		1.f, 1.f, 1.f,
	}, queue);
	mom.apply(weight, del_weight, queue);
	queue.finish();
	CHECK_RANGE_EQUAL(neu::scalar, 3, weight, (
		0.99f, 0.99f, 0.99f
	));

	mom.apply(weight, del_weight, queue);
	queue.finish();
	CHECK_RANGE_EQUAL(neu::scalar, 3, weight, (
		0.971f, 0.971f, 0.971f
	));
}

BOOST_AUTO_TEST_SUITE_END()
