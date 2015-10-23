#ifndef NEU_ACTIVATION_FUNC_SIGMOID_LOSS_HPP
#define NEU_ACTIVATION_FUNC_SIGMOID_LOSS_HPP
//20150528
#include <cmath>
#include <neu/as_const.hpp>
#include <neu/basic_type.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, sigmoid_loss_kernel, (float x), {
		return 1./(1.+exp(-x));
	});
	class sigmoid_loss {
	public:
		sigmoid_loss(std::size_t input_dim, std::size_t batch_size)
			: output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& x) {
			boost::compute::transform(x.begin(), x.end(),
				output_.begin(), neu::sigmoid_loss_kernel);
			return as_const(output_);
		}
	private:
		gpu_vector output_;
	};
	template<>
	class derivative<sigmoid_loss> {
	public:
		derivative(std::size_t input_dim, std::size_t batch_size)
				: output_(input_dim*batch_size) {
			boost::compute::fill(output_.begin(), output_.end(), 1.f);
		}
	
		decltype(auto) operator()(neu::gpu_vector) const {
			return (output_);
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SIGMOID_LOSS_HPP
