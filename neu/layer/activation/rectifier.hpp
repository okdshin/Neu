#ifndef NEU_LAYER_ACTIVATION_RECTIFIER_HPP
#define NEU_LAYER_ACTIVATION_RECTIFIER_HPP
//20151212
#include <neu/layer/activation/impl.hpp>
#include <neu/activation_func/rectifier.hpp>
namespace neu {
	namespace layer {
		using rectifier = activation<neu::rectifier>;

		namespace traits {
			template<>
			class save<rectifier> {
			public:
				static decltype(auto) call(rectifier const& ac,
						YAML::Emitter& emitter,
						boost::compute::command_queue&) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "rectifier"
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
		decltype(auto) load_rectifier(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "rectifier");
			return activation<neu::rectifier>(
				node["input_dim"].as<std::size_t>(),
				node["batch_size"].as<std::size_t>()
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_RECTIFIER_HPP
