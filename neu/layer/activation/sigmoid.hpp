#ifndef NEU_LAYER_ACTIVATION_SIGMOID_HPP
#define NEU_LAYER_ACTIVATION_SIGMOID_HPP
//20151212
#include <neu/layer/activation/impl.hpp>
#include <neu/activation_func/sigmoid.hpp>
namespace neu {
	namespace layer {
		using sigmoid = activation<neu::sigmoid>;

		decltype(auto) make_sigmoid(
				int input_dim, int batch_size) {
			return make_activation(input_dim, batch_size,
				neu::sigmoid());
		}

		namespace traits {
			template<>
			class serialize<sigmoid> {
			public:
				static decltype(auto) call(sigmoid const& ac,
						YAML::Emitter& emitter,
						boost::compute::command_queue&) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "sigmoid"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(ac)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(ac)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(ac)
					<< YAML::EndMap;
				}
			};
		}
		decltype(auto) deserialize_sigmoid(YAML::Node const& node,
				boost::compute::command_queue&) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "sigmoid");
			return activation<neu::sigmoid>(
				node["input_dim"].as<int>(),
				node["batch_size"].as<int>()
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_SIGMOID_HPP
