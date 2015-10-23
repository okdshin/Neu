#ifndef NEU_LAYER_YAML_HPP
#define NEU_LAYER_YAML_HPP
//20151023
#include <yaml-cpp/yaml.h>
#include <neu/basic_type.hpp>
#include <neu/fully_connected_layer.hpp>
#include <neu/convolution_layer.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/layer.hpp>
namespace neu_yaml_io_traits {
	template<typename T>
	class save_to_yaml {};

	template<typename T>
	class load_from_yaml {};
}
namespace neu {
	template<typename T>
	decltype(auto) save_to_yaml(YAML::Emitter& out, T const& t) {
		return neu_yaml_io_traits::save_to_yaml<T>::call(out, t);
	}
	using neu_yaml_io_traits::load_from_yaml;
}
namespace neu_yaml_io_traits {
	template<>
	class save_to_yaml<neu::cpu_vector> {
	public:
		static decltype(auto) call(YAML::Emitter& out, neu::cpu_vector const& vec) {
			out << YAML::Flow << YAML::BeginSeq;
			for(auto val : vec) {
				out << val;
			}
			out << YAML::EndSeq;
			return (out);
		}
	};
	template<>
	class load_from_yaml<neu::cpu_vector> {
	public:
		static decltype(auto) call(YAML::Node const& in) {
			Expects(in.IsSequence());
			neu::cpu_vector v(in.size());
			for (auto i = 0u; i < in.size(); i++) {
				  v[i] = in[i].as<neu::scalar>();
			}
			return v;
		}
	};

	template<>
	class save_to_yaml<neu::weight_decay_and_momentum> {
	public:
		static decltype(auto) call(YAML::Emitter& out,
				neu::weight_decay_and_momentum const& wd_m) {
			out << YAML::BeginMap;
				out << YAML::Key << "type"
					<< YAML::Value << "weight_decay_and_momentum";

				out << YAML::Key << "param" << YAML::Value;
				out << YAML::BeginMap;
					out << YAML::Key << "learning_rate"
						<< YAML::Value << wd_m.learning_rate();
					out << YAML::Key << "momentum_rate"
						<< YAML::Value << wd_m.momentum_rate();
					out << YAML::Key << "decay_rate" << YAML::Value << wd_m.decay_rate();
					out << YAML::Key << "delta_weight" << YAML::Value;
					neu::save_to_yaml(out, wd_m.delta_weight());
					out << YAML::Key << "delta_bias" << YAML::Value;
					neu::save_to_yaml(out, wd_m.delta_bias());
				out << YAML::EndMap;
			out << YAML::EndMap;
			return (out);
		}
	};
	template<>
	class load_from_yaml<neu::weight_decay_and_momentum> {
	public:
		static decltype(auto) call(YAML::Node const& in) {
			Expects(in["type"].as<std::string>() == "weight_decay_and_momentum");
			auto learning_rate = in["param"]["learning_rate"].as<float>();
			auto momentum_rate = in["param"]["momentum_rate"].as<float>();
			auto decay_rate = in["param"]["decay_rate"].as<float>();
			auto delta_weight =
				neu::load_from_yaml<neu::cpu_vector>::call(in["param"]["delta_weight"]);
			auto delta_bias =
				neu::load_from_yaml<neu::cpu_vector>::call(in["param"]["delta_bias"]);
			return neu::weight_decay_and_momentum(
				learning_rate, momentum_rate, decay_rate, delta_weight, delta_bias);
		}
	};

	template<>
	class save_to_yaml<neu::learning_rate_gen> {
	public:
		static decltype(auto) call(YAML::Emitter& out, neu::learning_rate_gen const& lrg) {
			std::cout << "layer as yaml!!!" << std::endl;
			if(auto* wd_m = lrg.target<neu::weight_decay_and_momentum>()) {
				neu::save_to_yaml(out, *wd_m);
			}
			else {
				out << YAML::BeginMap;
					out << YAML::Key << "learning_rate_gen" << YAML::Value << "unknown";
				out << YAML::EndMap;
			}
			return (out);
		}
	};

	template<typename LearningRateGen>
	class save_to_yaml<neu::fully_connected_layer<LearningRateGen>> {
	public:
		static decltype(auto) call(
			YAML::Emitter& out,
			neu::fully_connected_layer<LearningRateGen> const& fc
		) {
			out << YAML::BeginMap;
				out << YAML::Key << "type" << YAML::Value << "fully_connected";

				out << YAML::Key << "param" << YAML::Value;
				out << YAML::BeginMap;
					out << YAML::Key << "input_dim" << YAML::Value << fc.input_dim();
					out << YAML::Key << "batch_size" << YAML::Value << fc.batch_size();
					out << YAML::Key << "output_dim" << YAML::Value << fc.output_dim();
					out << YAML::Key << "weight" << YAML::Value;
					out << YAML::BeginMap;
						out << YAML::Key << "init" << YAML::Value << "fixed";
						out << YAML::Key << "param" << YAML::Value;
						out << YAML::BeginMap;
							out << YAML::Key << "value" << YAML::Value;
							neu::save_to_yaml(out, fc.weight());
						out << YAML::EndMap;
					out << YAML::EndMap;
					out << YAML::Key << "bias" << YAML::Value;
					out << YAML::BeginMap;
						out << YAML::Key << "init" << YAML::Value << "fixed";
						out << YAML::Key << "param" << YAML::Value;
						out << YAML::BeginMap;
							out << YAML::Key << "value" << YAML::Value;
							neu::save_to_yaml(out, fc.bias());
						out << YAML::EndMap;
					out << YAML::EndMap;
					out << YAML::Key << "learning_rate_gen" << YAML::Value;
					neu::save_to_yaml(out, fc.learning_rate_gen());
				out << YAML::EndMap;
			out << YAML::EndMap;
			return (out);
		}
	};

	template<typename LearningRateGen>
	class save_to_yaml<neu::convolution_layer<LearningRateGen>> {
	public:
		static decltype(auto) call(
			YAML::Emitter& out,
			neu::convolution_layer<LearningRateGen> const& conv
		) {
			out << YAML::BeginMap;
				out << YAML::Key << "layer" << YAML::Value << "convolution";

				out << YAML::Key << "param" << YAML::Value;
				out << YAML::BeginMap;
					out << YAML::Key << "input_width" << YAML::Value << conv.input_dim();
					out << YAML::Key << "batch_size" << YAML::Value << conv.batch_size();
					out << YAML::Key << "output_width" << YAML::Value << conv.output_dim();
					out << YAML::Key << "filter_width" << YAML::Value << conv.output_dim();
					out << YAML::Key << "filter" << YAML::Value;
					out << YAML::BeginMap;
						out << YAML::Key << "init" << YAML::Value << "fixed";
						out << YAML::Key << "param" << YAML::Value;
						out << YAML::BeginMap;
							out << YAML::Key << "value" << YAML::Value;
							neu::save_to_yaml(out, conv.filter());
						out << YAML::EndMap;
					out << YAML::EndMap;
					out << YAML::Key << "bias" << YAML::Value;
					out << YAML::BeginMap;
						out << YAML::Key << "init" << YAML::Value << "fixed";
						out << YAML::Key << "param" << YAML::Value;
						out << YAML::BeginMap;
							out << YAML::Key << "value" << YAML::Value;
							neu::save_to_yaml(out, conv.bias());
						out << YAML::EndMap;
					out << YAML::EndMap;
					//<< YAML::Key << "learning_rate_gen" << YAML::Value << fc.learning_rate_gen()
				out << YAML::EndMap;
			out << YAML::EndMap;
			return (out);
		}
	};
	template<>
	class save_to_yaml<neu::layer> {
	public:
		static decltype(auto) call(YAML::Emitter& out, neu::layer const& l) {
			std::cout << "layer as yaml!!!" << std::endl;
			if(auto* fc = l.target<neu::fully_connected_layer<>>()) {
				neu::save_to_yaml(out, *fc);
			}
			else {
				out << YAML::BeginMap;
					out << YAML::Key << "layer" << YAML::Value << "unknown";
				out << YAML::EndMap;
			}
			return (out);
		}
	};
}
namespace neu {

	/*
	decltype(auto) make_fully_connected_layer_from_yaml(YAML::Node const& node) {
		Expects(node["layer"].as<std::string>() == "fully_connected");
		auto input_dim = data["params"]["input_dim"].as<int>();
		auto batch_size = data["params"]["batch_size"].as<int>();
		auto output_dim = data["params"]["output_dim"].as<int>();
		if(data["params"]["initialization"]["weight"]["type"].as<std::string()
			== "constant") {
		}
		if(data["params"]["initialization"]["weight"]["type"].as<std::string()
			== "gaussian") {
		}
		if(data["params"]["initialization"]["weight"]["type"].as<std::string()
			== "fixed") {
		}

	}

	decltype(auto) make_layer_from_yaml(std::string const& yaml_str) {
		auto data = YAML::Load(yaml_str);

		if(data["name"]) {
			//TODO
		}
		if(data["layer"].as<std::string>() == "fully_connected") {
			if(data["params"]["initialization"]["weight"]["type"].as<std::string()
				== "gaussian") {
			}
			return make_fully_connected_layer<learning_rate_gen>(
				data["params"]["output_dim"].as<int>());
		}
	}
	*/
}// namespace neu

#endif //NEU_LAYER_YAML_HPP
