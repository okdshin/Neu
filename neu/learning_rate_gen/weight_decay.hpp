#ifndef NEU_WEIGHT_DECAY_HPP
#define NEU_WEIGHT_DECAY_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	/*
	template<typename LearningRateGen>
	class weight_decay {
	public:
		explicit weight_decay(scalar rate, LearningRateGen const& lrg) : rate_(rate) {}
		decltype(auto) operator()(gpu_vector const& weight, gpu_vector const& bias,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias,
				gpu_vector& new_delta_weight, gpu_vector& new_delta_bias) {
			using boost::compute::lambda::_1;
			boost::compute::transform(delta_weight.begin(), delta_weight.end(),
				decay_weight.begin(), delta_weight.begin(), boost::compute::minus<scalar>());
			boost::compute::transform(delta_bias.begin(), delta_bias.end(),
				decay_bias, delta_bias.begin(), boost::compute::minus<scalar>());
			lrg_(weight, bias, delta_weight, delta_bias, new_delta_weight, new_delta_bias);
		}
	private:
		scalar rate_;
	};
	*/
}// namespace neu

#endif //NEU_WEIGHT_DECAY_HPP
