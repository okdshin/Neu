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
			gpu_vector&& delta_weight, gpu_vector&& delta_bias)
			: learning_rate_(learning_rate),
			momentum_rate_(momentum_rate),
			decay_rate_(decay_rate),
			delta_weight_(std::move(delta_weight)),
			delta_bias_(std::move(delta_bias)) {}

		decltype(auto) learning_rate() const { return learning_rate_; }
		decltype(auto) momentum_rate() const { return momentum_rate_; }
		decltype(auto) decay_rate() const { return decay_rate_; }
		//decltype(auto) delta_weight() const { return to_cpu_vector(delta_weight_); }
		//decltype(auto) delta_bias() const { return to_cpu_vector(delta_bias_); }

		decltype(auto) operator()(
			gpu_vector& weight, gpu_vector& bias,
			gpu_vector const& del_weight, gpu_vector const& del_bias,
			boost::compute::command_queue& queue
		) {
			using boost::compute::lambda::_1;
			using boost::compute::lambda::_2;
			using boost::compute::transform;

			// momentum
			transform(delta_weight_.begin(), delta_weight_.end(), del_weight.begin(),
				delta_weight_.begin(), momentum_rate_*_1-learning_rate_*_2, queue);
			transform(delta_bias_.begin(), delta_bias_.end(), del_bias.begin(),
				delta_bias_.begin(), momentum_rate_*_1-learning_rate_*_2, queue);
			// weight decay (no for bias)
			transform(delta_weight_.begin(), delta_weight_.end(), weight.begin(),
				delta_weight_.begin(), _1-decay_rate_*_2, queue);

			// update
			transform(weight.begin(), weight.end(), delta_weight_.begin(),
				weight.begin(), _1+_2, queue);
			transform(bias.begin(), bias.end(), delta_bias_.begin(),
				bias.begin(), _1+_2, queue);
		}
	private:
		scalar learning_rate_, momentum_rate_, decay_rate_;
		gpu_vector delta_weight_, delta_bias_;
	};
	decltype(auto) make_weight_decay_and_momentum(
			float learning_rate, float momentum_rate, float decay_rate,
			std::size_t weight_dim, std::size_t bias_dim,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		gpu_vector delta_weight(weight_dim, 0.f, queue);
		gpu_vector delta_bias(bias_dim, 0.f, queue);
		return weight_decay_and_momentum(
			learning_rate, momentum_rate, decay_rate,
			std::move(delta_weight), std::move(delta_bias));
	}
}// namespace neu

#endif //NEU_WEIGHT_DECAY_AND_MOMENTUM_HPP
