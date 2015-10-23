#ifndef NEU_ACTIVATION_FUNC_SIGMOID_HPP
#define NEU_ACTIVATION_FUNC_SIGMOID_HPP
//20150528
#include <cmath>
#include <neu/as_const.hpp>
#include <neu/basic_type.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, sigmoid_kernel, (float x), {
		return 1./(1.+exp(-x));
	});
	BOOST_COMPUTE_FUNCTION(float, diff_sigmoid_kernel, (float x), {
		const float sigma = 1./(1.+exp(-x));
		return sigma*(1.-sigma);
	});
	class sigmoid {
	public:
		sigmoid(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& x) {
			boost::compute::transform(x.begin(), x.end(),
				output_.begin(), neu::sigmoid_kernel);
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
	template<>
	class derivative<sigmoid> {
	public:
		derivative(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector x) {
			boost::compute::transform(x.begin(), x.end(),
				output_.begin(), neu::diff_sigmoid_kernel);
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SIGMOID_HPP
