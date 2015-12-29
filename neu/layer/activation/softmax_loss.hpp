#ifndef NEU_LAYER_ACTIVATION_SOFTMAX_LOSS_HPP
#define NEU_LAYER_ACTIVATION_SOFTMAX_LOSS_HPP
//20151212
#include <neu/layer/activation/impl.hpp>
#include <neu/activation_func/softmax_loss.hpp>
namespace neu {
	namespace layer {
		using softmax_loss = activation<neu::softmax_loss>;

		decltype(auto) make_softmax_loss(
				int input_dim, int batch_size,
				boost::compute::context const& context) {
			return make_activation(input_dim, batch_size,
				neu::softmax_loss(input_dim, batch_size, context));
		}
		namespace traits {
			template<>
			class serialize<softmax_loss> {
			public:
				static decltype(auto) call(softmax_loss const& ac,
						YAML::Emitter& emitter,
						boost::compute::command_queue&) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "softmax_loss"
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
		decltype(auto) deserialize_softmax_loss(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "softmax_loss");
			return make_softmax_loss(
				node["input_dim"].as<int>(),
				node["batch_size"].as<int>(),
				queue.get_context()
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_SOFTMAX_LOSS_HPP
