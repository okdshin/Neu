#ifndef NEU_AVERAGE_POOLING_LAYER_PARAMETER_HPP
#define NEU_AVERAGE_POOLING_LAYER_PARAMETER_HPP
//20151026
#include <neu/layer_parameter.hpp>
#include <neu/average_pooling_layer/impl.hpp>
namespace neu {
	class average_pooling_layer_parameter {
		NEU_PP_PARAMETER(input_width)
		NEU_PP_PARAMETER(filter_width)
		NEU_PP_PARAMETER(input_channel_num)
		NEU_PP_PARAMETER(stride)
		NEU_PP_PARAMETER(pad)
		NEU_PP_PARAMETER(batch_size)
	public:
		decltype(auto) output_width() const {
			return (input_width()-filter_width()+1+2*pad())/stride();
		}
		decltype(auto) output_channel_num() const {
			return input_channel_num();
		}
		decltype(auto) output_dim() const {
			return output_width()*output_width()*output_channel_num();
		}
	};
	template<typename Params>
	decltype(auto) make_average_pooling_layer_parameter(Params const& params) {
		average_pooling_layer_parameter p;
		p.input_width(params.output_width());
		p.input_channel_num(params.output_channel_num());
		p.batch_size(params.batch_size());
		return p;
	}
	decltype(auto) make_average_pooling_layer(
			average_pooling_layer_parameter const& params,
			gpu_vector const& filter,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()){
		return make_average_pooling_layer(
			params.input_width(), params.output_width(), params.filter_width(),
			params.input_channel_num(), params.stride(), params.pad(),
			params.batch_size(), filter, queue.get_context());
	}
	decltype(auto) make_uniform_average_pooling_layer(
			average_pooling_layer_parameter const& params,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()){
		auto filter_dim = params.filter_width()*params.filter_width();
		gpu_vector filter(filter_dim, 1.f/filter_dim, queue);
		return make_average_pooling_layer(params, filter, queue);
	}
}// namespace neu

#endif //NEU_AVERAGE_POOLING_LAYER_PARAMETER_HPP
