#ifndef NEU_ACTIVATION_LAYER_PARAMETER_HPP
#define NEU_ACTIVATION_LAYER_PARAMETER_HPP
//20151025
#include <neu/layer_parameter.hpp>
#include <neu/activation_layer/impl.hpp>
namespace neu {
	class activation_layer_parameter {
		NEU_PP_PARAMETER(input_dim)
		NEU_PP_PARAMETER(output_dim)
		NEU_PP_PARAMETER(batch_size)
	};
	template<typename Param>
	decltype(auto) make_activation_layer_parameter(Param const& param) {
		activation_layer_parameter p;
		p.input_dim(param.output_dim());
		p.output_dim(param.output_dim());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename ActivationFunc>
	decltype(auto) make_activation_layer(activation_layer_parameter const& param,
			ActivationFunc const& activation_func) {
		return make_activation_layer<ActivationFunc>(
			param.input_dim(), param.output_dim(), param.batch_size(), activation_func);
	}
}// namespace neu

#endif //NEU_ACTIVATION_LAYER_PARAMETER_HPP
