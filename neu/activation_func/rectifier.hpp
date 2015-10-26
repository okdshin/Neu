#ifndef NEU_ACTIVATION_FUNC_RECTIFIER_HPP
#define NEU_ACTIVATION_FUNC_RECTIFIER_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/as_const.hpp>
#include <neu/basic_type.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, rectifier_kernel, (float x), {
		return x > 0 ? x : 0;
	});
	BOOST_COMPUTE_FUNCTION(float, diff_rectifier_kernel, (float x), {
		return x > 0 ? 1 : 0;
	});
	class rectifier {
	public:
		rectifier(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& input) {
			boost::compute::transform(input.begin(), input.end(),
				output_.begin(), neu::rectifier_kernel);
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
	template<>
	class derivative<rectifier> {
	public:
		derivative(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& input) {
			NEU_ASSERT(output_.size() == input.size());
			boost::compute::transform(input.begin(), input.end(),
				output_.begin(), neu::diff_rectifier_kernel);
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_RECTIFIER_HPP
