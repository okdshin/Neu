#ifndef NEU_OPTIMIZER_LOAD_HPP
#define NEU_OPTIMIZER_LOAD_HPP
//20151212
#include <yaml-cpp/yaml.h>
#include <neu/optimizer/any_optimizer.hpp>
#include <neu/optimizer/momentum.hpp>
namespace neu {
	namespace optimizer {
		decltype(auto) load(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto ot = node["optimizer_type"].as<std::string>();
			if(ot == "momentum") {
				return static_cast<any_optimizer>(load_momentum(node, queue));
			}
			else {
				throw "optimizer load error";
			}
		}
	}
}// namespace neu

#endif //NEU_OPTIMIZER_LOAD_HPP
