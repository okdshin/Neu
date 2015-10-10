#ifndef NEU_FIXED_LEARNING_RATE_GEN_HPP
#define NEU_FIXED_LEARNING_RATE_GEN_HPP
//20150830
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
namespace neu {
	const char fixed_learning_rate_gen_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void fixed_learning_rate_gen(
			__global float* weight, const __global float* delta_weight, const float rate)
		{
			const int o = get_global_id(0);
			weight[o] -= rate*delta_weight[o];
		}
	);
	class fixed_learning_rate_gen {
	public:
		explicit fixed_learning_rate_gen(scalar rate) :
			kernel_(make_kernel(fixed_learning_rate_gen_kernel_source,
				"fixed_learning_rate_gen")),
			rate_(rate) {}
		decltype(auto) operator()(gpu_vector& weight, gpu_vector& bias,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias) {
			auto weight_event = async_execute_nd_range_kernel<1>(
				kernel_, {0}, {weight.size()}, weight, delta_weight, rate_);
			auto bias_event = async_execute_nd_range_kernel<1>(
				kernel_, {0}, {bias.size()}, bias, delta_bias, rate_);
			weight_event.wait();
			bias_event.wait();
		}
	private:
		boost::compute::kernel kernel_;
		scalar rate_;
	};
}// namespace neu

#endif //NEU_FIXED_LEARNING_RATE_GEN_HPP
