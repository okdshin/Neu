#ifndef NEU_FIXED_LEARNING_RATE_HPP
#define NEU_FIXED_LEARNING_RATE_HPP
//20150830
#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <yaml-cpp/yaml.h>
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	namespace optimizer {
		class fixed_learning_rate {
		public:
			fixed_learning_rate() = default;
			fixed_learning_rate(float learning_rate)
				: learning_rate_(learning_rate) {}

			decltype(auto) learning_rate() const { return learning_rate_; }

			decltype(auto) apply(gpu_vector& weight, gpu_vector const& del_weight,
					boost::compute::command_queue& queue) {
				using boost::compute::lambda::_1;
				using boost::compute::lambda::_2;
				using boost::compute::transform;

				// weight -= del_weight
				transform(weight.begin(), weight.end(), del_weight.begin(),
					weight.begin(), _1-learning_rate_*_2, queue);
			}

			decltype(auto) serialize(YAML::Emitter& emitter,
					boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "optimizer_type"
						<< YAML::Value << "fixed_learning_rate"
					<< YAML::Key << "learning_rate"
						<< YAML::Value << learning_rate_
				<< YAML::EndMap;
			}

		private:
			scalar learning_rate_;
		};

		decltype(auto) deserialize_fixed_learning_rate(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["optimizer_type"].as<std::string>() == "fixed_learning_rate");
			return fixed_learning_rate(node["learning_rate"].as<float>());
		}
	}
}// namespace neu

#endif //NEU_FIXED_LEARNING_RATE_HPP
