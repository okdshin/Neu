#ifndef NEU_LAYER_DESERIALIZE_FWD_HPP
#define NEU_LAYER_DESERIALIZE_FWD_HPP
//20151215
#include <yaml-cpp/yaml.h>
#include <neu/layer/any_layer.hpp>
namespace neu {
	namespace layer {
		any_layer deserialize(YAML::Node const& node,
			boost::compute::command_queue& queue);
	}
}// namespace neu

#endif //NEU_LAYER_DESERIALIZE_FWD_HPP
