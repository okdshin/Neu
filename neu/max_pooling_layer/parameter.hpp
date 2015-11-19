#ifndef NEU_MAX_POOLING_LAYER_PARAMETER_HPP
#define NEU_MAX_POOLING_LAYER_PARAMETER_HPP
//20151026
#include <neu/layer_parameter.hpp>
#include <neu/max_pooling_layer/impl.hpp>
namespace neu {
	class max_pooling_layer_parameter {
		NEU_PP_PARAMETER(input_width)
		NEU_PP_PARAMETER(input_channel_num)
		NEU_PP_PARAMETER(batch_size)
		NEU_PP_PARAMETER(filter_width)
		NEU_PP_PARAMETER(stride)
		NEU_PP_PARAMETER(pad)
	public:
		decltype(auto) output_channel_num() const {
			return input_channel_num();
		}
		decltype(auto) output_width() const {
			return (input_width()-filter_width()+1+2*pad())/stride();
		}
		decltype(auto) output_dim() const {
			return output_width()*output_width()*output_channel_num();
		}
	};
	template<typename Param>
	decltype(auto) make_max_pooling_layer_parameter(Param const& param) {
		max_pooling_layer_parameter p;
		p.input_width(param.output_width());
		p.input_channel_num(param.output_channel_num());
		p.batch_size(param.batch_size());
		return p;
	}

	decltype(auto) make_max_pooling_layer(
		max_pooling_layer_parameter const& param,
		boost::compute::context const& context
			=boost::compute::system::default_context()
	){
		return make_max_pooling_layer(
			param.input_width(), param.output_width(), param.filter_width(),
			param.input_channel_num(), param.stride(),
			param.batch_size(), context);
	}
}// namespace neu

#endif //NEU_MAX_POOLING_LAYER_PARAMETER_HPP
