#ifndef NEU_OPTIMIZER_DESERIALIZE_HPP
#define NEU_OPTIMIZER_DESERIALIZE_HPP
//20151212
#include <exception>
#include <yaml-cpp/yaml.h>
#include <neu/optimizer/any_optimizer.hpp>
#include <neu/optimizer/momentum.hpp>
namespace neu {
	namespace optimizer {
		class deserialize_error : public std::exception {
		public:
			explicit deserialize_error(std::string const& optimizer_id) : optimizer_id_(optimizer_id) {}

			virtual const char* what() const noexcept {
				return ("Unknown optimizer \""+ optimizer_id_
					+ "\" is tried to be deserialized. Check optimizer data and deserialize function.").c_str();
			}
		private:
			std::string optimizer_id_;
		};

		decltype(auto) deserialize(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto ot = node["optimizer_type"].as<std::string>();
			if(ot == "momentum") {
				return static_cast<any_optimizer>(deserialize_momentum(node, queue));
			} else
			if(ot == "fixed_learning_rate") {
				return static_cast<any_optimizer>(deserialize_fixed_learning_rate(node, queue));
			}
			else {
				throw deserialize_error(ot);
			}
		}
	}
}// namespace neu

#endif //NEU_OPTIMIZER_DESERIALIZE_HPP
