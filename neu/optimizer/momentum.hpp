#ifndef NEU_MOMENTUM_HPP
#define NEU_MOMENTUM_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <yaml-cpp/yaml.h>
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	namespace optimizer {
		class momentum {
		public:
			momentum() = default;
			momentum(
				scalar learning_rate,
				scalar momentum_rate,
				scalar weight_decay,
				cpu_vector const& delta_weight,
				boost::compute::command_queue& queue)
				: learning_rate_(learning_rate),
				momentum_rate_(momentum_rate),
				weight_decay_(weight_decay),
				delta_weight_(delta_weight.begin(), delta_weight.end(), queue) {}

			momentum(
				scalar learning_rate,
				scalar momentum_rate,
				scalar weight_decay,
				int weight_dim,
				boost::compute::command_queue& queue)
			: momentum(learning_rate, momentum_rate, weight_decay, cpu_vector(weight_dim, 0.f), queue) {}

			decltype(auto) learning_rate() const { return learning_rate_; }
			decltype(auto) momentum_rate() const { return momentum_rate_; }
			decltype(auto) weight_decay() const { return weight_decay_; }
			decltype(auto) delta_weight(boost::compute::command_queue& queue) const {
				return to_cpu_vector(delta_weight_, queue); }

			decltype(auto) apply(gpu_vector& weight, gpu_vector const& del_weight,
					boost::compute::command_queue& queue) {
				using boost::compute::lambda::_1;
				using boost::compute::lambda::_2;
				using boost::compute::transform;

				// momentum
				transform(delta_weight_.begin(), delta_weight_.end(), del_weight.begin(),
					delta_weight_.begin(), momentum_rate_*_1-learning_rate_*_2, queue);

				// weight -= weight_decay*weight
				transform(weight.begin(), weight.end(), weight.begin(),
					weight.begin(), _1-learning_rate_*weight_decay_*_2, queue);

				// weight += delta_weight
				transform(weight.begin(), weight.end(), delta_weight_.begin(),
					weight.begin(), _1+_2, queue);
			}

			decltype(auto) serialize(YAML::Emitter& emitter,
					boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "optimizer_type"
						<< YAML::Value << "momentum"
					<< YAML::Key << "learning_rate"
						<< YAML::Value << learning_rate_
					<< YAML::Key << "momentum_rate"
						<< YAML::Value << momentum_rate_
					<< YAML::Key << "weight_decay"
						<< YAML::Value << weight_decay_
#ifndef NEU_LAYER_SERIALIZE_WITHOUT_LONG_VECTOR
					<< YAML::Key << "delta_weight"
						<< YAML::Value << YAML::Flow << delta_weight(queue)
#endif
				<< YAML::EndMap;
			}

		private:
			scalar learning_rate_, momentum_rate_, weight_decay_;
			gpu_vector delta_weight_;
		};

		decltype(auto) deserialize_momentum(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["optimizer_type"].as<std::string>() == "momentum");
			return momentum(
				node["learning_rate"].as<scalar>(),
				node["momentum_rate"].as<scalar>(),
				node["weight_decay"].as<scalar>(),
				node["delta_weight"].as<cpu_vector>(),queue);
		}
	}
}// namespace neu

#endif //NEU_MOMENTUM_HPP
