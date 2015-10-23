#ifndef NEU_SUBTRACT_DELTA_WEIGHT_HPP
#define NEU_SUBTRACT_DELTA_WEIGHT_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	template<typename LearningRateGen>
	class subtract_delta_weight {
	public:
		explicit subtract_delta_weight(scalar weight_rate, scalar bias_rate,
				LearningRateGen const& lrg)
			: weight_rate_(weight_rate), bias_rate_(bias_rate), lrg_(lrg) {}

		decltype(auto) operator()(gpu_vector& weight, gpu_vector& bias,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias) {
			using boost::compute::lambda::_1;

			gpu_vector new_delta_weight(delta_weight.size());
			gpu_vector new_delta_bias(delta_bias.size());
			lrg_(weight, bias, delta_weight, delta_bias, new_delta_weight, new_delta_bias);

			boost::compute::transform(new_delta_weight.begin(), new_delta_weight.end(),
				new_delta_weight.begin(), _1*weight_rate_);
			boost::compute::transform(weight.begin(), weight.end(), new_delta_weight.begin(),
				weight.begin(), boost::compute::minus<scalar>());

			boost::compute::transform(new_delta_bias.begin(), new_delta_bias.end(),
				new_delta_bias.begin(), _1*bias_rate_);
			boost::compute::transform(bias.begin(), bias.end(), new_delta_bias.begin(),
				bias.begin(), boost::compute::minus<scalar>());
		}
	private:
		scalar weight_rate_, bias_rate_;
		LearningRateGen lrg_;
	};
	template<typename LearningRateGen>
	decltype(auto) make_subtract_delta_weight(
			scalar weight_rate, scalar bias_rate, LearningRateGen const& lrg) {
		return subtract_delta_weight<LearningRateGen>(weight_rate, bias_rate, lrg);
	}
	template<typename LearningRateGen>
	decltype(auto) make_subtract_delta_weight(LearningRateGen const& lrg) {
		return subtract_delta_weight<LearningRateGen>(1.f, 1.f, lrg);
	}
}// namespace neu

#endif //NEU_SUBTRACT_DELTA_WEIGHT_HPP
