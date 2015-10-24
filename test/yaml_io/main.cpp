#include <iostream>
#include <neu/fully_connected_layer.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/layer_yaml.hpp>

class test {};
namespace neu_yaml_io_traits {
	template<>
	class save_to_yaml<test> {
	public:
		static decltype(auto) call(YAML::Emitter& out, test) {
			return out << "test";
		}
	};
}

int main() {
	std::cout << "hello world" << std::endl;
	std::mt19937 rand(0);
	auto input_dim = 2u;
	auto batch_size = 4u;
	neu::scalar base_lr = 0.01;
	neu::scalar momentum = 0.0;
	neu::scalar weight_decay = 0.;
	auto fc1_param = neu::fully_connected_layer_parameter()
		.input_dim(input_dim).batch_size(batch_size)
		.output_dim(10);
	auto fc12_g = [&rand, dist=std::normal_distribution<>(0.f, 1.f)]
		() mutable { return dist(rand); };
	auto constant_g = [](){ return 0.f; };
	neu::learning_rate_gen lrg =
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			fc1_param.weight_dim(), fc1_param.bias_dim());
	auto fc1 = neu::make_fully_connected_layer(fc1_param, fc12_g, constant_g, lrg);
	{
		YAML::Emitter out;
		neu::save_to_yaml(out, fc1);
		std::cout << "fc\n" << out.c_str() << std::endl;
	}
	{
		YAML::Emitter out;
		neu::layer lfc1 = fc1;
		neu::save_to_yaml(out, lfc1);
		std::cout << "lfc\n" << out.c_str() << std::endl;
	}
	{
		YAML::Emitter out;
		neu::save_to_yaml(out, test());
		std::cout << "test\n" << out.c_str() << std::endl;
	}

	{
		neu::learning_rate_gen wd_m = neu::make_weight_decay_and_momentum(
			base_lr, momentum, weight_decay, 20, 10);
		YAML::Emitter out;
		neu::save_to_yaml(out, wd_m);
		std::cout << "weight_decay_and_momentum\n" << out.c_str() << std::endl;
		auto in = YAML::Load(out.c_str());
		auto wd_m_loaded = neu::load_from_yaml<neu::weight_decay_and_momentum>(in);
		{
			YAML::Emitter out;
			neu::save_to_yaml(out, wd_m_loaded);
			std::cout << "weight_decay_and_momentum\n" << out.c_str() << std::endl;
		}
	}
}
