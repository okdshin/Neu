#ifndef NEU_AVERAGE_POOLING_LAYER_HPP
#define NEU_AVERAGE_POOLING_LAYER_HPP
//20150622
#include <gsl.h>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameter.hpp>
#include <neu/convolution_layer/indices.hpp>
namespace neu {
	const char average_pooling_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void average_pooling(
			const __global int* indices_range_list_for_output,
			const __global int* input_indices_list_for_output,
			const __global int* filter_indices_list_for_output,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num,
			const __global float* input, __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(2);
			const int k = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.0;
			const int indices_begin = indices_range_list_for_output[i];
			const int indices_end = indices_range_list_for_output[i+1];
			for(int j = indices_begin; j < indices_end; ++j) {
				const int filter_index = index(filter_indices_list_for_output[j],
					0, 0, filter_width, input_channel_num);
				const int input_index = index(input_indices_list_for_output[j],
					k, b, input_width, input_channel_num);
				sum += filter[filter_index]*input[input_index];
			}
			const int output_index = index(i, k, b, output_width, input_channel_num);
			output[output_index] = sum;
		}
	);
	const char average_pooling_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void average_pooling_back(
			const __global int* indices_range_list_for_input,
			const __global int* output_indices_list_for_input,
			const __global int* filter_indices_list_for_input,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num,
			__global float* input, const __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(2);
			const int k = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.0;
			const int indices_begin = indices_range_list_for_input[i];
			const int indices_end = indices_range_list_for_input[i+1];
			for(int j = indices_begin; j < indices_end; ++j) {
				const int filter_index = index(filter_indices_list_for_input[j],
					0, 0, filter_width, input_channel_num);
				const int output_index = index(output_indices_list_for_input[j],
					k, b, output_width, input_channel_num);
				sum += filter[filter_index]*output[output_index];
			}
			const int input_index = index(i, k, b, input_width, input_channel_num);
			input[input_index] = sum;
		}
	);
	
	class average_pooling_layer {
	public:
		average_pooling_layer(std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num, std::size_t stride, std::size_t batch_size,
			convolution_indices const& indices, gpu_vector const& filter,
			kernel const& pooling_kernel, kernel const& pooling_back_kernel)
			: input_width_(input_width), filter_width_(filter_width),  
			input_channel_num_(input_channel_num),
			stride_(stride), batch_size_(batch_size),
			indices_(indices), filter_(filter),
			pooling_kernel_(pooling_kernel),
			pooling_back_kernel_(pooling_back_kernel),
			output_width_(output_width),
			next_input_(output_width_*output_width_*input_channel_num*batch_size_),
			prev_delta_(input_width_*input_width_*input_channel_num*batch_size)
			{}

		decltype(auto) forward(gpu_vector const& input) {
			Expects(is_all_of_finite(input));
			execute_nd_range_kernel<3>(pooling_kernel_,
				{0, 0, 0}, {output_width_*output_width_, input_channel_num_, batch_size_},
				indices_.indices_range_list_for_output,
				indices_.input_indices_list_for_output,
				indices_.filter_indices_list_for_output,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				input, next_input_, filter_);
			Ensures(is_all_of_finite(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(is_all_of_finite(delta));
			execute_nd_range_kernel<3>(pooling_back_kernel_,
				{0, 0, 0}, {input_width_*input_width_, input_channel_num_, batch_size_},
				indices_.indices_range_list_for_input,
				indices_.output_indices_list_for_input,
				indices_.filter_indices_list_for_input,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				prev_delta_, delta, filter_);
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t stride_;
		std::size_t batch_size_;

		convolution_indices indices_;

		gpu_vector filter_;

		kernel pooling_kernel_;
		kernel pooling_back_kernel_;

		std::size_t output_width_;

		gpu_vector next_input_;
		gpu_vector prev_delta_;
	};
	decltype(auto) make_average_pooling_layer(
		std::size_t input_width, std::size_t output_width, std::size_t filter_width,
		std::size_t input_channel_num, std::size_t stride, std::size_t pad,
		std::size_t batch_size, 
		gpu_vector const& filter
	){
		auto indices = neu::make_convolution_indices(
			input_width, output_width, filter_width, stride, pad);
		auto pooling_kernel
			= make_kernel(average_pooling_kernel_source, "average_pooling");
		auto pooling_back_kernel
			= make_kernel(average_pooling_back_kernel_source, "average_pooling_back");
		return average_pooling_layer(input_width, output_width, filter_width,
			input_channel_num, stride, batch_size, indices, filter,
			pooling_kernel, pooling_back_kernel);
	}

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
			gpu_vector const& filter) {
		return make_average_pooling_layer(
			params.input_width(), params.output_width(), params.filter_width(),
			params.input_channel_num(), params.stride(), params.pad(),
			params.batch_size(), filter);
	}
	decltype(auto) make_uniform_average_pooling_layer(
			average_pooling_layer_parameter const& params){
		auto filter_dim = params.filter_width()*params.filter_width();
		gpu_vector filter(filter_dim);
		boost::compute::fill(filter.begin(), filter.end(), (1.f/filter_dim));
		return make_average_pooling_layer(params, filter);
	}
}// namespace neu

#endif //NEU_AVERAGE_POOLING_LAYER_HPP
