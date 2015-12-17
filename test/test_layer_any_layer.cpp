#define BOOST_TEST_MODULE TestSoftmaxLoss
#include <boost/test/unit_test.hpp>

#include <neu/layer/any_layer.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

class test_layerA {
public:
	bool operator==(test_layerA const&) const { return true; }

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

class test_layerB {
public:
	int value = 0;
	bool operator==(test_layerB const& other) const { return value == other.value; }

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

BOOST_AUTO_TEST_CASE(comparison) {
	{
		neu::layer::any_layer al;
		BOOST_CHECK(al == al);
	}
	{
		neu::layer::any_layer la = test_layerA();
		BOOST_CHECK(la == la);
	}
	{
		neu::layer::any_layer la1 = test_layerA();
		neu::layer::any_layer la2 = test_layerA();
		BOOST_CHECK(la1 == la2);
	}
	{
		neu::layer::any_layer la = test_layerA();
		neu::layer::any_layer lb = test_layerB();
		BOOST_CHECK(la != lb);
	}
	{
		neu::layer::any_layer lb1 = test_layerB();
		neu::layer::any_layer lb2 = test_layerB{42};
		BOOST_CHECK(lb1 != lb2);
	}
}

BOOST_AUTO_TEST_SUITE_END()
