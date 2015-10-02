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
	//TODO bias
	const char convolution_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void convolution(
			const __global float* input, __global float* output,
			const __global float* filter, const int stride,
			const int input_channel_num, const int output_channel_num,
			const int input_width, const int output_width, const int filter_width)
		{
			const int b = get_global_id(2);
			const int gr = get_global_id(1);
			const int gc = get_global_id(0);

			for(int m = 0; m < output_channel_num; ++m) {
				float sum = 0.0;
				for(int k = 0; k < input_channel_num; ++k) {
					for(int fr = 0; fr < filter_width; ++fr) {
						for(int fc = 0; fc < filter_width; ++fc) {
							const int ir = gr*stride-filter_width/2+fr;
							const int ic = gc*stride-filter_width/2+fc;
							if(0 <= ir && ir < input_width && 0 <= ic && ic < input_width) {
								const int input_index = index(ic, ir, k, b,
									input_width, input_channel_num);
								const int filter_index = index(fc, fr, k, m,
									filter_width, input_channel_num);
								sum += input[input_index]*filter[filter_index];
							}
						}
					}
				}
				const int output_index = index(gc, gr, m, b, output_width, output_channel_num);
				output[output_index] = sum;
			}
		}
	);

	//TODO check
	const char convolution_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void convolution_back(
			__global float* input, const __global float* output, const __global float* filter,
			const int stride, const int input_channel_num, const int output_channel_num,
			const int input_width, const int output_width, const int filter_width)
		{
			const int b = get_global_id(2);
			const int gr = get_global_id(1);
			const int gc = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int m = 0; m < output_channel_num; ++m) {
					for(int fr = 0; fr < filter_width; ++fr) {
						for(int fc = 0; fc < filter_width; ++fc) {
							const int or = (gr+filter_width/2-fr)/stride;
							const int oc = (gc+filter_width/2-fc)/stride;
							if((gr+filter_width/2-fr)%stride == 0
									&& (gc+filter_width/2-fc)%stride == 0
									&& 0 <= oc && oc < output_width
									&& 0 <= or && or < output_width) {
								const int output_index = index(oc, or, m, b,
									output_width, output_channel_num);
								const int filter_index = index(fc, fr, k, m,
									filter_width, input_channel_num);
								sum += output[output_index]*filter[filter_index];
							}
						}
					}
				}
				const int input_index = index(gc, gr, k, b,
					input_width, input_channel_num);
				input[input_index] = sum;
			}
		}
	);
	const char update_delta_filters_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void update_delta_filters(
			const __global float * input,
			const __global float * delta,
			__global float * delta_filters,
			const int stride,
			const int input_channel_num,
			const int output_channel_num,
			const int input_width,
			const int output_width,
			const int filter_width,
			const int batch_size)
		{
			const int m = get_global_id(2);
			const int fr = get_global_id(1);
			const int fc = get_global_id(0);
			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int b = 0; b < batch_size; ++b) {
					for(int or = 0; or < output_width; ++or) {
						for(int oc = 0; oc < output_width; ++oc) {
							const int ir = or*stride-filter_width/2+fr;
							const int ic = oc*stride-filter_width/2+fc;
							if(0 <= ir && ir < input_width 
									&& 0 <= ic && ic < input_width) {
								const int input_index = index(ic, ir, k, b,
									input_width, input_channel_num);
								const int output_index = index(oc, or, m, b,
									output_width, output_channel_num);
								sum += input[input_index]*delta[output_index];
							}
						}
					}
				}
				const int filter_index =
					index(fc, fr, k, m, filter_width, input_channel_num);
				delta_filters[filter_index] = sum;
			}
		}
	);
	template<typename LearningRateGen>
	class convolution_layer {
	public:
		convolution_layer(
			std::size_t input_width, std::size_t filter_width,
			std::size_t input_channel_num, std::size_t output_channel_num,
			std::size_t stride, std::size_t batch_size,
			gpu_vector const& filters, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			kernel const& convolution_kernel,
			kernel const& convolution_back_kernel,
			kernel const& update_delta_filters_kernel)
		: input_width_(input_width), filter_width_(filter_width),
		input_channel_num_(input_channel_num), output_channel_num_(output_channel_num),
		stride_(stride), batch_size_(batch_size),
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
		delta_bias_(bias_.size())
		{
			boost::compute::fill(delta_filters_.begin(), delta_filters_.end(), 0.f);
		}

		decltype(auto) get_filters() const { return (filters_); }

		decltype(auto) forward(gpu_vector const& input) {
			assert(input.size() == input_.size());
			auto future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			neu::execute_nd_range_kernel<3>(convolution_kernel_,
				{0,0,0}, {output_width_, output_width_, batch_size_},
				input, next_input_, filters_,
				static_cast<int>(stride_), static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_), static_cast<int>(input_width_),
				static_cast<int>(output_width_), static_cast<int>(filter_width_));
			future.wait();
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			assert(delta.size() == delta_.size());
			auto future = boost::compute::
				copy_async(delta.begin(), delta.end(), delta_.begin());
			neu::execute_nd_range_kernel<3>(convolution_back_kernel_, 
				{0, 0, 0}, {input_width_, input_width_, batch_size_},
				prev_delta_, delta, filters_,
				static_cast<int>(stride_), static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_), static_cast<int>(input_width_),
				static_cast<int>(output_width_), static_cast<int>(filter_width_));
			future.wait();
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() {
			neu::execute_nd_range_kernel<3>(update_delta_filters_kernel_,
				{0, 0, 0}, {filter_width_, filter_width_, output_channel_num_},
				input_, delta_, delta_filters_,
				static_cast<int>(stride_), static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_), static_cast<int>(input_width_),
				static_cast<int>(output_width_), static_cast<int>(filter_width_),
				static_cast<int>(batch_size_));
			learning_rate_gen_(filters_, bias_, delta_filters_, delta_bias_);
			boost::compute::fill(delta_filters_.begin(), delta_filters_.end(), 0.f);
		}
		decltype(auto) get_delta_filters() const { return (delta_filters_); }

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t output_channel_num_;
		std::size_t stride_;
		std::size_t batch_size_;

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
