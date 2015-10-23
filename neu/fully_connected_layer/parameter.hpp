#ifndef NEU_FULLY_CONNECTED_LAYER_PARAMETER_HPP
#define NEU_FULLY_CONNECTED_LAYER_PARAMETER_HPP
//20151023
#include <neu/layer_parameter.hpp>
#include <neu/fully_connected_layer/fully_connected_layer_impl.hpp>
namespace neu {
	class fully_connected_layer_parameter {
		NEU_PP_PARAMETER(input_dim)
		NEU_PP_PARAMETER(output_dim)
		NEU_PP_PARAMETER(batch_size)
	public:
		decltype(auto) weight_dim() const {
			return input_dim()*output_dim();
		}
		decltype(auto) bias_dim() const {
			return output_dim();
		}
	};
	template<typename Param>
	decltype(auto) make_fully_connected_layer_parameter(Param const& param) {
		fully_connected_layer_parameter p;
		p.input_dim(param.output_dim());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			fully_connected_layer_parameter const& param,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		return make_fully_connected_layer(
			param.input_dim(), param.output_dim(), param.batch_size(),
			weight, bias, learning_rate_gen);
	}
	template<typename WeightGen, typename BiasGen, typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			fully_connected_layer_parameter const& param,
			WeightGen const& wg, BiasGen const& bg,
			LearningRateGen const& learning_rate_gen) {
		return make_fully_connected_layer(
			param,
			neu::make_random_gpu_vector(param.weight_dim(), wg),
			neu::make_random_gpu_vector(param.bias_dim(), bg),
			learning_rate_gen);
	}
}// namespace neu

#endif //NEU_FULLY_CONNECTED_LAYER_PARAMETER_HPP
