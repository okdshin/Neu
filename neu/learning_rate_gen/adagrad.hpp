#ifndef NEU_ADAGRAD_HPP
#define NEU_ADAGRAD_HPP
//20150830

namespace neu {
	const char adagrad_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void adagrad(
			__global float* weight, const __global float* delta_weight,
			__global float* weight_r, const float rate)
		{
			const int o = get_global_id(0);
			weight_r[o] += delta_weight[o]*delta_weight[o];
			weight[o] -= (rate/(1.f+sqrt(weight_r[o])))*delta_weight[o];
		}
	);
	class adagrad {
	public:
		adagrad(std::size_t weight_size, std::size_t bias_size, scalar rate) :
			kernel_(make_kernel(adagrad_kernel_source,"adagrad")),
			rate_(rate), weight_r_(weight_size), bias_r_(bias_size)
		{
			boost::compute::fill(weight_r_.begin(), weight_r_.end(), 0.f);
			boost::compute::fill(bias_r_.begin(), bias_r_.end(), 0.f);
		}
		decltype(auto) operator()(gpu_vector& weight, gpu_vector& bias,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias) {
			assert(weight_r_.size() == delta_weight.size());
			assert(bias_r_.size() == delta_bias.size());
			auto weight_event = async_execute_nd_range_kernel<1>(kernel_, {0}, {weight.size()},
				weight, delta_weight, weight_r_, rate_);
			auto bias_event = async_execute_nd_range_kernel<1>(kernel_, {0}, {bias.size()},
				bias, delta_bias, bias_r_, rate_);
			weight_event.wait();
			bias_event.wait();
		}
	private:
		boost::compute::kernel kernel_;
		scalar rate_;
		gpu_vector weight_r_;
		gpu_vector bias_r_;
	};
}// namespace neu

#endif //NEU_ADAGRAD_HPP
