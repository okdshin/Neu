#ifndef NEU_FIXED_LEARNING_RATE_GEN_HPP
#define NEU_FIXED_LEARNING_RATE_GEN_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	class fixed_learning_rate_gen {
	public:
		explicit fixed_learning_rate_gen(scalar rate) : rate_(rate) {}
		decltype(auto) operator()(gpu_vector, gpu_vector,
				gpu_vector const& delta_weight, gpu_vector const& delta_bias,
				gpu_vector& new_delta_weight, gpu_vector& new_delta_bias) {
			using boost::compute::lambda::_1;
			boost::compute::transform(delta_weight.begin(), delta_weight.end(),
				new_delta_weight.begin(), _1*rate_);
			boost::compute::transform(delta_bias.begin(), delta_bias.end(),
				new_delta_bias.begin(), _1*rate_);
			boost::compute::system::default_queue().finish();
		}
	private:
		scalar rate_;
	};
}// namespace neu

#endif //NEU_FIXED_LEARNING_RATE_GEN_HPP
