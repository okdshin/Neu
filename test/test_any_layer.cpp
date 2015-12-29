#define BOOST_TEST_MODULE TestAnyLayer
#include <boost/test/unit_test.hpp>

#include <neu/layer/any_layer.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

class test_layerA {
public:
	decltype(auto) input_rank() const { return 1; }
	decltype(auto) output_rank() const { return 1; }
	decltype(auto) input_size(neu::layer::rank_id ri) const {
		return ri == neu::layer::rank_id::dim ? 42 : 0; }
	decltype(auto) output_size(neu::layer::rank_id ri) const {
		return ri == neu::layer::rank_id::dim ? 666 : 0; }
	decltype(auto) batch_size() const { return 101; }

	void test_forward(std::size_t batch_size,
			neu::range::gpu_vector_range const& input,
			neu::range::gpu_vector_range const& output,
			boost::compute::command_queue& queue) {
	}
	void forward(neu::range::gpu_vector_range const& input,
			neu::range::gpu_vector_range const& output,
			boost::compute::command_queue& queue) {
	}

	void backward_top(neu::range::gpu_vector_range const& delta,
			boost::compute::command_queue& queue) {
	}
	void backward(neu::range::gpu_vector_range const& delta,
			neu::range::gpu_vector_range const& prev_delta,
			boost::compute::command_queue& queue) {
	}

	void update(boost::compute::command_queue& queue) {
	}

	void serialize(YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
	}
};

BOOST_AUTO_TEST_CASE(nullptr_comparison) {
	neu::layer::any_layer la_null;
	BOOST_CHECK(la_null == nullptr);
	BOOST_CHECK(nullptr == la_null);
	BOOST_CHECK(!(la_null != nullptr));
	BOOST_CHECK(!(nullptr != la_null));

	neu::layer::any_layer la = test_layerA();
	BOOST_CHECK(!(la == nullptr));
	BOOST_CHECK(!(nullptr == la));
	BOOST_CHECK(la != nullptr);
	BOOST_CHECK(nullptr != la);
}

BOOST_AUTO_TEST_SUITE_END()
