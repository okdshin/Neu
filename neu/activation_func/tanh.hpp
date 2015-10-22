#ifndef NEU_ACTIVATION_FUNC_TANH_HPP
#define NEU_ACTIVATION_FUNC_TANH_HPP
//20150528
#include <neu/as_const.hpp>
#include <neu/basic_type.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, tanh_kernel, (float x), {
		return tanh(x);
	});
	BOOST_COMPUTE_FUNCTION(float, diff_tanh_kernel, (float x), {
		const float t = tan(x);
		return 1.-t*t;
	});
	class tanh {
	public:
		tanh(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& input) {
			boost::compute::transform(input.begin(), input.end(),
				output_.begin(), neu::tanh_kernel);
			boost::compute::system::default_queue().finish();
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
	template<>
	class derivative<tanh> {
	public:
		derivative(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& input) {
			boost::compute::transform(input.begin(), input.end(),
				output_.begin(), neu::diff_tanh_kernel);
			boost::compute::system::default_queue().finish();
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_TANH_HPP
