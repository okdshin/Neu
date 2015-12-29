#ifndef NEU_LAYER_ACTIVATION_LEAKY_RECTIFIER_HPP
#define NEU_LAYER_ACTIVATION_LEAKY_RECTIFIER_HPP
//20151212
#include <neu/layer/activation/impl.hpp>
#include <neu/activation_func/leaky_rectifier.hpp>
namespace neu {
	namespace layer {
		using leaky_rectifier =
			activation<neu::leaky_rectifier, neu::derivative_leaky_rectifier>;

		decltype(auto) make_leaky_rectifier(
				int input_dim, int batch_size,
				neu::scalar negative_scale) {
			return make_activation(input_dim, batch_size,
				neu::leaky_rectifier(negative_scale),
				derivative_leaky_rectifier(negative_scale));
		}
		namespace traits {
			template<>
			class serialize<leaky_rectifier> {
			public:
				static decltype(auto) call(leaky_rectifier const& ac,
						YAML::Emitter& emitter,
						boost::compute::command_queue&) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "leaky_rectifier"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(ac)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(ac)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(ac)
						<< YAML::Key << "negative_scale"
							<< YAML::Value << ac.activation_func().negative_scale()
					<< YAML::EndMap;
				}
			};
		}
		decltype(auto) deserialize_leaky_rectifier(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "leaky_rectifier");
			return make_leaky_rectifier(
				node["input_dim"].as<int>(),
				node["batch_size"].as<int>(),
				node["negative_scale"].as<scalar>()
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_LEAKY_RECTIFIER_HPP
