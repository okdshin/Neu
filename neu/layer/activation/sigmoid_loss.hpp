#ifndef NEU_LAYER_ACTIVATION_SIGMOID_LOSS_HPP
#define NEU_LAYER_ACTIVATION_SIGMOID_LOSS_HPP
//20151212
#include <neu/layer/activation/impl.hpp>
#include <neu/activation_func/sigmoid_loss.hpp>
namespace neu {
	namespace layer {
		using sigmoid_loss = activation<neu::sigmoid_loss>;

		decltype(auto) make_sigmoid_loss(
				int input_dim, int batch_size) {
			return make_activation(input_dim, batch_size,
				neu::sigmoid_loss());
		}

		namespace traits {
			template<>
			class serialize<sigmoid_loss> {
			public:
				static decltype(auto) call(sigmoid_loss const& ac,
						YAML::Emitter& emitter,
						boost::compute::command_queue&) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "sigmoid_loss"
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
		decltype(auto) deserialize_sigmoid_loss(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "sigmoid_loss");
			return activation<neu::sigmoid_loss>(
				node["input_dim"].as<int>(),
				node["batch_size"].as<int>()
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_SIGMOID_LOSS_HPP
