#ifndef NEU_MOMENTUM_HPP
#define NEU_MOMENTUM_HPP
//20150830
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
namespace neu {
	const char momentum_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void momentum(
			__global float* weight, const __global float* delta_weight, const float rate)
		{
			const int o = get_global_id(0);
			weight[o] -= rate*delta_weight[o];
		}
	);
	template<typename LearningRateGen>
	class momentum {
	public:
		explicit momentum(scalar rate, LearningRateGen const& lrg)
			: kernel_(make_kernel(momentum_kernel_source,"momentum")), rate_(rate), lrg_(lrg) {}
		decltype(auto) operator()(gpu_vector& weight, gpu_vector& bias,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias) {
			learning_rate_gen_(weight, bias, delta_weight, delta_bias);
			//TODO
			/*
			auto weight_event = async_execute_nd_range_kernel<1>(
				kernel_, {0}, {weight.size()}, weight, delta_weight, rate_);
			auto bias_event = async_execute_nd_range_kernel<1>(
				kernel_, {0}, {bias.size()}, bias, delta_bias, rate_);
			weight_event.wait();
			bias_event.wait();
			*/
		}
	private:
		boost::compute::kernel kernel_;
		scalar rate_;
		learning_rate_gen_;
	};
}// namespace neu

#endif //NEU_MOMENTUM_HPP
