#ifndef NEU_CONVOLUTION_LAYER_HPP
#define NEU_CONVOLUTION_LAYER_HPP
//20150619
#include <boost/compute/algorithm.hpp>
#include <boost/compute/algorithm/generate.hpp>
#include <boost/compute/kernel.hpp>
#include <neu/basic_type.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameters.hpp>
namespace neu {

	decltype(auto) calc_indices(int input_width, int filter_width, int stride) {
		const auto half_filter_width = filter_width/2;
		const auto output_width = input_width/stride;
		std::vector<cpu_indices> filter_indices_list_for_input(input_width*input_width);
		std::vector<cpu_indices> output_indices_list_for_input(input_width*input_width);
		std::vector<cpu_indices> filter_indices_list_for_output(output_width*output_width);
		std::vector<cpu_indices> input_indices_list_for_output(output_width*output_width);
		std::vector<cpu_indices> input_indices_list_for_filter(filter_width*filter_width);
		std::vector<cpu_indices> output_indices_list_for_filter(filter_width*filter_width);
		for(auto _or = 0; _or < output_width; ++_or) {
			for(auto oc = 0; oc < output_width; ++oc) {
				for(auto fr = 0; fr < filter_width; ++fr) {
					for(auto fc = 0; fc < filter_width; ++fc) {
						const auto ir = _or*stride-half_filter_width+fr;
						const auto ic = oc*stride-half_filter_width+fc;

						const auto input_index = ir*input_width+ic;
						const auto output_index = _or*output_width+oc;
						const auto filter_index = fr*filter_width+fc;
						if(0 <= ir && ir < input_width && 0 <= ic && ic < input_width) {
							filter_indices_list_for_output[output_index].push_back(filter_index);
							input_indices_list_for_output[output_index].push_back(input_index);

							filter_indices_list_for_input[input_index].push_back(filter_index);
							output_indices_list_for_input[input_index].push_back(output_index);

							input_indices_list_for_filter[filter_index].push_back(input_index);
							output_indices_list_for_filter[filter_index].push_back(output_index);
						}
					}
				}
			}
		}
		return std::make_tuple(
			output_width,
			input_indices_list_for_output, filter_indices_list_for_output,
			output_indices_list_for_input, filter_indices_list_for_input,
			input_indices_list_for_filter, output_indices_list_for_filter);
	}
	decltype(auto) concat_indices(std::vector<cpu_indices> const& indices_list) {
		cpu_indices concatinated;
		for(auto const& indices : indices_list) {
			concatinated.insert(concatinated.end(), indices.begin(), indices.end());
		}
		return concatinated;
	}
	decltype(auto) make_range_list(std::vector<std::vector<int>> const& indices_list) {
		cpu_indices range_list;
		range_list.push_back(0);
		auto index = 0u;
		for(auto const& indices : indices_list) {
			index += indices.size();
			range_list.push_back(index);
		}
		return range_list;
	}
	//TODO bias
	const char convolution_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void convolution(
			const __global int* indices_range_list_for_output,
			const __global int* input_indices_list_for_output,
			const __global int* filter_indices_list_for_output,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num, const int output_channel_num,
			const __global float* input, __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			for(int m = 0; m < output_channel_num; ++m) {
				float sum = 0.0;
				for(int k = 0; k < input_channel_num; ++k) {
					const int indices_begin = indices_range_list_for_output[i];
					const int indices_end = indices_range_list_for_output[i+1];
					for(int j = indices_begin; j < indices_end; ++j) {
						const int filter_index = index(filter_indices_list_for_output[j],
							k, m, filter_width, input_channel_num);
						const int input_index = index(input_indices_list_for_output[j],
							k, b, input_width, input_channel_num);
						sum += filter[filter_index]*input[input_index];
					}
				}
				const int output_index = index(i, m, b, output_width, output_channel_num);
				output[output_index] = sum;
			}
		}
	);

	const char convolution_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void convolution_back(
			const __global int* indices_range_list_for_input,
			const __global int* output_indices_list_for_input,
			const __global int* filter_indices_list_for_input,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num, const int output_channel_num,
			__global float* input, const __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int m = 0; m < output_channel_num; ++m) {
					const int indices_begin = indices_range_list_for_input[i];
					const int indices_end = indices_range_list_for_input[i+1];
					for(int j = indices_begin; j < indices_end; ++j) {
						const int filter_index = index(filter_indices_list_for_input[j],
							k, m, filter_width, input_channel_num);
						const int output_index = index(output_indices_list_for_input[j],
							m, b, output_width, output_channel_num);
						sum += filter[filter_index]*output[output_index];
					}
				}
				const int input_index = index(i, k, b, input_width, input_channel_num);
				input[input_index] = sum;
			}
		}
	);
	const char update_delta_filters_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void update_delta_filters(
			const __global int* indices_range_list_for_filter,
			const __global int* input_indices_list_for_filter,
			const __global int* output_indices_list_for_filter,
			const int input_width, const int output_width,
			const int filter_width, const int batch_size,
			const int input_channel_num, const int output_channel_num,
			const __global float* input, const __global float* output,
			__global float* filter)
		{
			const int m = get_global_id(1);
			const int i = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int b = 0; b < batch_size; ++b) {
					const int indices_begin = indices_range_list_for_filter[i];
					const int indices_end = indices_range_list_for_filter[i+1];
					for(int j = indices_begin; j < indices_end; ++j) {
						const int input_index = index(input_indices_list_for_filter[j],
							k, b, input_width, input_channel_num);
						const int output_index = index(output_indices_list_for_filter[j],
							m, b, output_width, output_channel_num);
						sum += input[input_index]*output[output_index];
					}
				}
				const int filter_index = index(i, k, m, filter_width, input_channel_num);
				filter[filter_index] = sum;
			}
		}
	);
	template<typename LearningRateGen>
	class convolution_layer {
	public:
		convolution_layer(
			std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num, std::size_t output_channel_num,
			std::vector<cpu_indices> const& filter_indices_list_for_output,
			std::vector<cpu_indices> const& input_indices_list_for_output,
			std::vector<cpu_indices> const& filter_indices_list_for_input,
			std::vector<cpu_indices> const& output_indices_list_for_input,
			std::vector<cpu_indices> const& input_indices_list_for_filter,
			std::vector<cpu_indices> const& output_indices_list_for_filter,
			std::size_t stride, std::size_t batch_size,
			gpu_vector const& filters, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			kernel const& convolution_kernel,
			kernel const& convolution_back_kernel,
			kernel const& update_delta_filters_kernel)
		: input_width_(input_width), filter_width_(filter_width),
		input_channel_num_(input_channel_num), output_channel_num_(output_channel_num),
		stride_(stride), batch_size_(batch_size),
		filter_indices_list_for_output_(filter_indices_list_output),
		input_indices_list_for_output_(input_indices_list_output),
		filter_indices_list_for_input_(filter_indices_list_input),
		output_indices_list_for_input_(output_indices_list_input),
		input_indices_list_for_filter_(input_indices_list_filter),
		output_indices_list_for_filter_(output_indices_list_filter),
		filters_(filters), bias_(bias),
		learning_rate_gen_(learning_rate_gen),
		convolution_kernel_(convolution_kernel),
		convolution_back_kernel_(convolution_back_kernel),
		update_delta_filters_kernel_(update_delta_filters_kernel),
		output_width_(input_width/stride),
		input_(input_width_*input_width_*input_channel_num_*batch_size_),
		next_input_(output_width_*output_width_*output_channel_num_*batch_size_),
		delta_(next_input_.size()),
		prev_delta_(input_.size()),
		delta_filters_(filters_.size()),
		delta_bias_(bias_.size()) {}

		decltype(auto) get_filters() const { return (filters_); }

		decltype(auto) forward(gpu_vector const& input) {
			assert(input.size() == input_.size());
			auto input_copy_future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			neu::execute_nd_range_kernel<2>(convolution_kernel_,
				{0, 0}, {output_width_*output_width_, batch_size_},
				indices_range_list_for_output_, input_indices_list_output_,
				filter_indices_list_output_,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				input, next_input_, filters_);
			input_copy_future.wait();
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			assert(delta.size() == delta_.size());
			auto delta_copy_future = boost::compute::
				copy_async(delta.begin(), delta.end(), delta_.begin());
			neu::execute_nd_range_kernel<2>(convolution_back_kernel,
				{0, 0}, {input_width_*input_width_, batch_size},
				indices_range_list_for_input_, output_indices_list_for_input_,
				filter_indices_list_for_input_,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				prev_delta_, delta, filters_);
			delta_copy_future.wait();
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() {
			neu::execute_nd_range_kernel<2>(update_delta_filters_kernel,
				{0, 0}, {filter_width_*filter_width_, output_channel_num_},
				indices_range_list_for_filter_, input_indices_list_for_filter_,
				output_indices_list_for_filter_,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_), static_cast<int>(batch_size_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				input_, delta_, delta_filters_);
		}
		decltype(auto) get_delta_filters() const { return (delta_filters_); }

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t output_channel_num_;
		std::size_t stride_;
		std::size_t batch_size_;

		std::vector<cpu_indices> filter_indices_list_for_output_;
		std::vector<cpu_indices> input_indices_list_for_output_;

		std::vector<cpu_indices> filter_indices_list_for_input_;
		std::vector<cpu_indices> output_indices_list_for_input_;

		std::vector<cpu_indices> input_indices_list_for_filter_;
		std::vector<cpu_indices> output_indices_list_for_filter_;

		gpu_vector filters_;
		gpu_vector bias_;

		LearningRateGen learning_rate_gen_;

		boost::compute::kernel convolution_kernel_;
		boost::compute::kernel convolution_back_kernel_;
		boost::compute::kernel update_delta_filters_kernel_;

		std::size_t output_width_;

		gpu_vector input_;
		gpu_vector next_input_;
		gpu_vector delta_;
		gpu_vector prev_delta_;

		gpu_vector delta_filters_;
		gpu_vector delta_bias_;
	};

	template<typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		std::size_t input_width,
		std::size_t filter_width,
		std::size_t input_channel_num,
		std::size_t output_channel_num,
		std::size_t stride,
		std::size_t batch_size,
		gpu_vector const& filters,
		gpu_vector const& bias,
		LearningRateGen const& learning_rate_gen,
		boost::compute::kernel const& conv_kernel
			=make_kernel(convolution_kernel_source, "convolution"),
		boost::compute::kernel const& conv_back_kernel
			=make_kernel(convolution_back_kernel_source, "convolution_back"),
		boost::compute::kernel const& update_delta_filters_kernel
			=make_kernel(update_delta_filters_kernel_source, "update_delta_filters")
	) {
		return convolution_layer<LearningRateGen>(
			input_width, filter_width, input_channel_num, output_channel_num,
			stride, batch_size, filters, bias, learning_rate_gen,
			conv_kernel, conv_back_kernel, update_delta_filters_kernel);
	}

	class convolution_layer_parameters {
		NEU_PP_PARAMETERS(input_width)
		NEU_PP_PARAMETERS(input_channel_num)
		NEU_PP_PARAMETERS(batch_size)
		NEU_PP_PARAMETERS(filter_width)
		NEU_PP_PARAMETERS(output_channel_num)
		NEU_PP_PARAMETERS(stride)
	public:
		convolution_layer_parameters() = default;
		template<typename Params>
		explicit convolution_layer_parameters(Params const& params) {
			input_width(params.output_width());
			input_channel_num(params.output_channel_num());
			batch_size(params.batch_size());
		}
		decltype(auto) output_width() const {
			return input_width()/stride();
		}
		decltype(auto) output_dim() const {
			return output_width()*output_width()*output_channel_num();
		}
	};
	template<typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		convolution_layer_parameters const& params,
		gpu_vector const& filters,
		gpu_vector const& bias,
		LearningRateGen const& learning_rate_gen,
		boost::compute::kernel const& conv_kernel
			=make_kernel(convolution_kernel_source, "convolution"),
		boost::compute::kernel const& conv_back_kernel
			=make_kernel(convolution_back_kernel_source, "convolution_back"),
		boost::compute::kernel const& update_delta_filters_kernel
			=make_kernel(update_delta_filters_kernel_source, "update_delta_filters")
	){
		return make_convolution_layer(
			params.input_width(), params.filter_width(),
			params.input_channel_num(), params.output_channel_num(),
			params.stride(), params.batch_size(),
			filters, bias,
			learning_rate_gen,
			conv_kernel, conv_back_kernel, update_delta_filters_kernel);
	}
	template<typename URNG, typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		convolution_layer_parameters const& params,
		URNG const& g,
		LearningRateGen const& learning_rate_gen,
		boost::compute::kernel const& conv_kernel
			=make_kernel(convolution_kernel_source, "convolution"),
		boost::compute::kernel const& conv_back_kernel
			=make_kernel(convolution_back_kernel_source, "convolution_back"),
		boost::compute::kernel const& update_delta_filters_kernel
			=make_kernel(update_delta_filters_kernel_source, "update_delta_filters")
	){
		return make_convolution_layer(
			params.input_width(), params.filter_width(),
			params.input_channel_num(), params.output_channel_num(),
			params.stride(), params.batch_size(),
			neu::make_random_gpu_vector(
				params.filter_width()*params.filter_width()
				*params.input_channel_num()*params.output_channel_num(), g),
			neu::make_random_gpu_vector(
				params.filter_width()*params.filter_width()
				*params.output_channel_num(), g),
			learning_rate_gen,
			conv_kernel, conv_back_kernel, update_delta_filters_kernel);
	}
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_HPP
