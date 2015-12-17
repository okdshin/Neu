#ifndef NEU_ACTIVATION_FUNC_LEAKY_RECTIFIER_HPP
#define NEU_ACTIVATION_FUNC_LEAKY_RECTIFIER_HPP
//20150528
#include <boost/compute/command_queue.hpp>
#include <boost/compute/functional.hpp>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, leaky_rectifier_kernel,
			(float x, float negative_scale), {
		return x > 0 ? x : negative_scale*x;
	});
	BOOST_COMPUTE_FUNCTION(float, derivative_leaky_rectifier_kernel,
			(float x, float negative_scale), {
		return x > 0 ? 1 : negative_scale;
	});
	class leaky_rectifier {
	public:
		leaky_rectifier() = default;
		explicit leaky_rectifier(scalar negative_scale)
			: negative_scale_(negative_scale) {}

		decltype(auto) negative_scale() const { return negative_scale_; }

		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			range::transform(input, output,
				boost::compute::bind(leaky_rectifier_kernel,
					boost::compute::placeholders::_1, negative_scale_), queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}
	
	private:
		scalar negative_scale_ = 0.f;
	};
	class derivative_leaky_rectifier {
	public:
		derivative_leaky_rectifier() = default;
		explicit derivative_leaky_rectifier(scalar negative_scale)
			: negative_scale_(negative_scale) {}

		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			range::transform(input, output,
				boost::compute::bind(derivative_leaky_rectifier_kernel,
					boost::compute::placeholders::_1, negative_scale_), queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}
	
	private:
		scalar negative_scale_ = 0.01f;
	};

}// namespace neu

#endif //NEU_ACTIVATION_FUNC_LEAKY_RECTIFIER_HPP
