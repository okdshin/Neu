#ifndef NEU_WEIGHT_DECAY_AND_MOMENTUM_HPP
#define NEU_WEIGHT_DECAY_AND_MOMENTUM_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	class weight_decay_and_momentum {
	public:
		weight_decay_and_momentum(
			float learning_rate, float momentum_rate, float decay_rate,
			std::size_t weight_dim, std::size_t bias_dim) 
			: learning_rate_(learning_rate),
			momentum_rate_(momentum_rate),
			decay_rate_(decay_rate),
			delta_weight_(weight_dim), delta_bias_(bias_dim)
		{
			boost::compute::fill(delta_weight_.begin(), delta_weight_.end(), 0.f);
			boost::compute::fill(delta_bias_.begin(), delta_bias_.end(), 0.f);
		}

		decltype(auto) operator()(gpu_vector& weight, gpu_vector& bias,
				gpu_vector const& del_weight, gpu_vector const& del_bias) {
			using boost::compute::lambda::_1;
			using boost::compute::lambda::_2;
			using boost::compute::transform;

			// momentum
			transform(delta_weight_.begin(), delta_weight_.end(), del_weight.begin(),
				delta_weight_.begin(), momentum_rate_*_1-learning_rate_*_2);
			transform(delta_bias_.begin(), delta_bias_.end(), del_bias.begin(),
				delta_bias_.begin(), momentum_rate_*_1-learning_rate_*_2);
			// weight decay (no for bias)
			transform(delta_weight_.begin(), delta_weight_.end(), weight.begin(),
				delta_weight_.begin(), _1-decay_rate_*_2);

			// update
			transform(weight.begin(), weight.end(), delta_weight_.begin(),
				weight.begin(), _1+_2);
			transform(bias.begin(), bias.end(), delta_bias_.begin(),
				bias.begin(), _1+_2);

			boost::compute::system::default_queue().finish();
		}
	private:
		scalar learning_rate_, momentum_rate_, decay_rate_;
		gpu_vector delta_weight_, delta_bias_;
	};
}// namespace neu

#endif //NEU_WEIGHT_DECAY_AND_MOMENTUM_HPP
