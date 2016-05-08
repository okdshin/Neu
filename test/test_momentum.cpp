#define BOOST_TEST_MODULE TestMomentum
#include <boost/test/unit_test.hpp>

#include <neu/optimizer/momentum.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>

#include "check_tool.hpp"
#include "context_setup.hpp"

namespace ct = neu_check_tool;

BOOST_AUTO_TEST_CASE(forward) {
	const auto base_lr = 0.01f;
	const auto mom_rate = 0.9f;
	const auto weight_decay = 0.0004f;
	const auto weight_dim = 3;
	neu::optimizer::momentum mom(base_lr, mom_rate, weight_decay, weight_dim, queue);
	neu::gpu_vector weight({
		1.f, 1.f, 1.f,
	}, queue);
	neu::gpu_vector del_weight({
		1.f, 1.f, 1.f,
	}, queue);
	mom.apply(weight, del_weight, queue);
	queue.finish();
	ct::check_range_close(weight, {
		0.99f, 0.99f, 0.99f
	}, 1.e-4f);

	mom.apply(weight, del_weight, queue);
	queue.finish();
	ct::check_range_close(weight, {
		0.971f, 0.971f, 0.971f
	}, 1.e-4f);
}

BOOST_AUTO_TEST_SUITE_END()
