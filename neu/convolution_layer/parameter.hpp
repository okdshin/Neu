#ifndef NEU_CONVOLUTION_LAYER_PARAMETER_HPP
#define NEU_CONVOLUTION_LAYER_PARAMETER_HPP
//20151005
#include <neu/layer_parameter.hpp>
#include <neu/convolution_layer/convolution_layer_impl.hpp>
#include <neu/convolution_layer/factory.hpp>
namespace neu {
	class convolution_layer_parameter {
		NEU_PP_PARAMETER(input_width)
		NEU_PP_PARAMETER(input_channel_num)
		NEU_PP_PARAMETER(batch_size)
		NEU_PP_PARAMETER(filter_width)
		NEU_PP_PARAMETER(output_channel_num)
		NEU_PP_PARAMETER(stride)
		NEU_PP_PARAMETER(pad)
	public:
		decltype(auto) output_width() const {
			return (input_width()-filter_width()+1+2*pad())/stride();
		}
		decltype(auto) output_dim() const {
			return output_width()*output_width()*output_channel_num();
		}
		decltype(auto) weight_dim() const {
			return filter_width()*filter_width()*input_channel_num()*output_channel_num();
		}
		decltype(auto) bias_dim() const {
			return filter_width()*filter_width()*output_channel_num();
		}
	};
	template<typename Param>
	decltype(auto) make_convolution_layer_parameter(Param const& param) {
		convolution_layer_parameter p;
		p.input_width(param.output_width());
		p.input_channel_num(param.output_channel_num());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		convolution_layer_parameter const& param,
		gpu_vector const& filters,
		gpu_vector const& bias,
		LearningRateGen const& learning_rate_gen
	){
		return make_convolution_layer(
			param.input_width(), param.output_width(), param.filter_width(),
			param.input_channel_num(), param.output_channel_num(),
			param.stride(), param.pad(), param.batch_size(),
			filters, bias,
			learning_rate_gen);
	}
	template<typename WeightGen, typename BiasGen, typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		convolution_layer_parameter const& param,
		WeightGen const& wg, BiasGen const& bg,
		LearningRateGen const& learning_rate_gen
	){
		return make_convolution_layer(
			param,
			neu::make_random_gpu_vector(param.weight_dim(), wg),
			neu::make_random_gpu_vector(param.bias_dim(), bg),
			learning_rate_gen);
	}
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_PARAMETER_HPP
