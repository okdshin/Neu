#define BOOST_TEST_MODULE TestSoftmaxLoss
#include <boost/test/unit_test.hpp>

#include <neu/layer/any_layer.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

class test_layerA {
public:
	std::size_t input_dim() const { return 42; }
	std::size_t output_dim() const { return 101; }
	std::size_t batch_size() const { return 666; }

	void test_forward(std::size_t batch_size,
			neu::range::gpu_vector_range const& input,
			neu::range::gpu_vector_range const& output,
			boost::compute::command_queue& queue) {
	}
	void forward(neu::range::gpu_vector_range const& input,
			neu::range::gpu_vector_range const& output,
			boost::compute::command_queue& queue) {
	}

	void backward(neu::range::gpu_vector_range const& delta,
			neu::range::gpu_vector_range const& prev_delta,
			bool is_top,
			boost::compute::command_queue& queue) {
	}

	void update(boost::compute::command_queue& queue) {
	}

	void save(YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
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
