#ifndef NEU_CONVOLUTION_LAYER_CONVOLUTION_LAYER_IMPL_HPP
#define NEU_CONVOLUTION_LAYER_CONVOLUTION_LAYER_IMPL_HPP
//20151005
#include <gsl.h>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/convolution_layer/indices.hpp>
namespace neu {
	template<typename LearningRateGen>
	class convolution_layer {
	public:
		convolution_layer(
			std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num, std::size_t output_channel_num,
			std::size_t stride, std::size_t pad, std::size_t batch_size,
			gpu_vector const& filters, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			convolution_indices const& indices,
			kernel const& convolution_kernel,
			kernel const& convolution_back_kernel,
			kernel const& update_del_filters_kernel)
		: input_width_(input_width), filter_width_(filter_width),
		input_channel_num_(input_channel_num), output_channel_num_(output_channel_num),
		stride_(stride), pad_(pad), batch_size_(batch_size),
		indices_(indices),
		filters_(filters), bias_(bias),
		learning_rate_gen_(learning_rate_gen),
		convolution_kernel_(convolution_kernel),
		convolution_back_kernel_(convolution_back_kernel),
		update_del_filters_kernel_(update_del_filters_kernel),
		output_width_(output_width),
		input_(input_width_*input_width_*input_channel_num_*batch_size_),
		next_input_(output_width_*output_width_*output_channel_num_*batch_size_),
		delta_(next_input_.size()),
		prev_delta_(input_.size()),
		del_filters_(filters.size()),
		del_bias_(bias.size()) {}

		decltype(auto) get_filters() const { return (filters_); }
		decltype(auto) get_bias() const { return (bias_); }

		decltype(auto) forward(gpu_vector const& input) {
			Expects(is_all_of_finite(input));
			auto input_copy_future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			neu::execute_nd_range_kernel<3>(convolution_kernel_,
				{0, 0, 0}, {output_width_, output_width_, batch_size_},
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				static_cast<int>(stride_), static_cast<int>(pad_),
				input, next_input_, filters_, bias_);
			/* // by indices version
			neu::execute_nd_range_kernel<2>(convolution_kernel_,
				{0, 0}, {output_width_*output_width_, batch_size_},
				indices_.indices_range_list_for_output,
				indices_.input_indices_list_for_output,
				indices_.filter_indices_list_for_output,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				input, next_input_, filters_, bias_);
			*/
			input_copy_future.wait();
			Ensures(is_all_of_finite(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(is_all_of_finite(delta));
			auto delta_copy_future = boost::compute::
				copy_async(delta.begin(), delta.end(), delta_.begin());
			neu::execute_nd_range_kernel<2>(convolution_back_kernel_,
				{0, 0}, {input_width_*input_width_, batch_size_},
				indices_.indices_range_list_for_input,
				indices_.output_indices_list_for_input,
				indices_.filter_indices_list_for_input,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				prev_delta_, delta, filters_);
			delta_copy_future.wait();
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() {
			neu::async_execute_nd_range_kernel<3>(update_del_filters_kernel_,
				{0, 0, 0}, {filter_width_, filter_width_, output_channel_num_},
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				static_cast<int>(stride_), static_cast<int>(pad_),
				static_cast<int>(batch_size_),
				input_, delta_, del_filters_, del_bias_);
			/* by indices version
			neu::execute_nd_range_kernel<2>(update_del_filters_kernel_,
				{0, 0}, {filter_width_*filter_width_, output_channel_num_},
				indices_.indices_range_list_for_filter,
				indices_.input_indices_list_for_filter,
				indices_.output_indices_list_for_filter,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_), static_cast<int>(batch_size_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				input_, delta_, del_filters_, del_bias_);
			*/
			Ensures(is_all_of_finite(del_filters_));
			Ensures(is_all_of_finite(del_bias_));
			learning_rate_gen_(filters_, bias_, del_filters_, del_bias_);
			Ensures(is_all_of_finite(filters_));
			Ensures(is_all_of_finite(bias_));
		}

		decltype(auto) print(std::ostream& os) {
			fmt::format("{{name: \"conv\", layer: convolution, params: {{}}}}");
		}
		decltype(auto) get_del_filters() const { return (del_filters_); }
		decltype(auto) get_del_bias() const { return (del_bias_); }

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t output_channel_num_;
		std::size_t stride_;
		std::size_t pad_;
		std::size_t batch_size_;

		convolution_indices indices_;

		gpu_vector filters_;
		gpu_vector bias_;

		LearningRateGen learning_rate_gen_;

		boost::compute::kernel convolution_kernel_;
		boost::compute::kernel convolution_back_kernel_;
		boost::compute::kernel update_del_filters_kernel_;

		std::size_t output_width_;

		gpu_vector input_;
		gpu_vector next_input_;
		gpu_vector delta_;
		gpu_vector prev_delta_;

		gpu_vector del_filters_;
		gpu_vector del_bias_;
	};
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_CONVOLUTION_LAYER_IMPL_HPP
