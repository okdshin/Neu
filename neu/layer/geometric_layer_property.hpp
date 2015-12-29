#ifndef NEU_LAYER_GEOMETRIC_LAYER_PROPERTY_HPP
#define NEU_LAYER_GEOMETRIC_LAYER_PROPERTY_HPP
//20151220
#include <yaml-cpp/yaml.h>
namespace neu {
	namespace layer {
		struct geometric_layer_property {
			int input_width;
			int filter_width;
			int input_channel_num;
			int output_channel_num;
			int stride;
			int pad;
		};

		decltype(auto) serialize(geometric_layer_property const& glp,
				YAML::Emitter& emitter){
			emitter 
				<< YAML::Key << "input_width" << YAML::Value << glp.input_width
				<< YAML::Key << "filter_width" << YAML::Value << glp.filter_width
				<< YAML::Key << "input_channel_num"
					<< YAML::Value << glp.input_channel_num
				<< YAML::Key << "output_channel_num"
					<< YAML::Value << glp.output_channel_num
				<< YAML::Key << "stride" << YAML::Value << glp.stride
				<< YAML::Key << "pad" << YAML::Value << glp.pad
			;
		}

		decltype(auto) deserialize_geometric_layer_property(YAML::Node const& node) {
			return geometric_layer_property{
				node["input_width"].as<int>(),
				node["filter_width"].as<int>(),
				node["input_channel_num"].as<int>(),
				node["output_channel_num"].as<int>(),
				node["stride"].as<int>(),
				node["pad"].as<int>()
			};
		}

		decltype(auto) output_width(geometric_layer_property const& glp) {
			return (glp.input_width-glp.filter_width+1+2*glp.pad)/glp.stride;
		}

		decltype(auto) input_dim(geometric_layer_property const& glp) {
			return glp.input_width*glp.input_width*glp.input_channel_num;
		}

		decltype(auto) output_dim(geometric_layer_property const& glp) {
			const auto ow = output_width(glp);
			return ow*ow*glp.output_channel_num;
		}

		decltype(auto) filters_size(geometric_layer_property const& glp) {
			return glp.filter_width*glp.filter_width
				*glp.input_channel_num*glp.output_channel_num;
		}

	}
}// namespace neu

#endif //NEU_LAYER_GEOMETRIC_LAYER_PROPERTY_HPP
