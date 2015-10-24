#ifndef NEU_MAX_POOLING_LAYER_HPP
#define NEU_MAX_POOLING_LAYER_HPP
//20150622
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameter.hpp>
namespace neu {
	const char max_pooling_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void max_pooling(const __global float* input, __global float* output,
			__global int* indices, const int stride, const int input_channel_num,
			const int input_width, const int output_width, const int filter_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float max_val = 0.0;
				int max_index = 0;
				const int r_start = max(0, or*stride-filter_width/2);
				const int r_end =
					min(input_width, or*stride-filter_width/2+filter_width);
				for(int r = r_start; r < r_end; ++r) {
					const int c_start = max(0, oc*stride-filter_width/2);
					const int c_end = 
						min(input_width, oc*stride-filter_width/2+filter_width);
					for(int c = c_start; c < c_end; ++c) {
						const int input_index = index(c, r, k, b,
							input_width, input_channel_num);
						const float input_val = input[input_index];
						if(max_val < input_val) {
							max_val = input_val;
							max_index = input_index;
						}
					}
				}
				const int output_index =
					index(oc, or, k, b, output_width, input_channel_num);
				output[output_index] = max_val;
				indices[output_index] = max_index;
			}
		}
	);
	
	class max_pooling_layer {
	public:
		max_pooling_layer(std::size_t input_width, std::size_t filter_width,
			std::size_t input_channel_num, std::size_t stride, std::size_t batch_size,
			kernel const& pooling_kernel)
			: input_width_(input_width), filter_width_(filter_width),  
			input_channel_num_(input_channel_num),
			stride_(stride), batch_size_(batch_size),
			pooling_kernel_(pooling_kernel),
			output_width_(input_width/stride),
			next_input_(output_width_*output_width_*input_channel_num*batch_size_),
			prev_delta_(input_width_*input_width_*input_channel_num*batch_size),
			gpu_indices_(next_input_.size()) {}

		decltype(auto) forward(gpu_vector const& input) {
			Expects(is_all_of_finite(input));
			auto event = enqueue_nd_range_kernel<3>(pooling_kernel_,
				{0, 0, 0}, {output_width_, output_width_, batch_size_},
				input, next_input_, gpu_indices_,
				static_cast<int>(stride_), static_cast<int>(input_channel_num_),
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_));
			event.wait(); // needless?
			indices_ = to_cpu_indices(gpu_indices_);
			Ensures(is_all_of_finite(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(is_all_of_finite(delta));
			cpu_vector cpu_prev_delta(
				input_width_*input_width_*input_channel_num_*batch_size_, 0);
			auto cpu_delta = to_cpu_vector(delta);
			for(auto i = 0u; i < indices_.size(); ++i) {
				assert(i < cpu_delta.size());
				assert(static_cast<std::size_t>(indices_[i]) < cpu_prev_delta.size());
				cpu_prev_delta[indices_[i]] += cpu_delta[i];
			}
			prev_delta_ = to_gpu_vector(cpu_prev_delta);
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() { /* do nothing */ }

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t stride_;
		std::size_t batch_size_;

		kernel pooling_kernel_;

		std::size_t output_width_;

		gpu_vector next_input_;
		std::vector<int> indices_;
		gpu_vector prev_delta_;

		gpu_indices gpu_indices_;
	};
	decltype(auto) make_max_pooling_layer(
		std::size_t input_width, std::size_t filter_width,
		std::size_t input_channel_num, std::size_t stride, std::size_t batch_size,
		kernel const& pooling_kernel=make_kernel(max_pooling_kernel_source, "max_pooling")
	){
		return max_pooling_layer(input_width, filter_width,
			input_channel_num, stride, batch_size, pooling_kernel);
	}

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
		kernel const& pooling_kernel=make_kernel(max_pooling_kernel_source, "max_pooling")
	){
		return make_max_pooling_layer(
			param.input_width(), param.filter_width(),
			param.input_channel_num(), param.stride(),
			param.batch_size(), pooling_kernel);
	}
}// namespace neu

#endif //NEU_MAX_POOLING_LAYER_HPP
