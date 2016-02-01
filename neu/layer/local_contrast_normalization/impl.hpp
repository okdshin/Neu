#ifndef NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_IMPL_HPP
#define NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_IMPL_HPP
//20151225
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/local_contrast_normalization/kernel_source.hpp>

namespace neu {
	namespace layer {
		class local_contrast_normalization {
		public:
			local_contrast_normalization(
				geometric_layer_property const& glp,
				int batch_size,
				scalar alpha, scalar beta,
				boost::compute::context const& context)
			: glp_(glp), batch_size_(batch_size),
			alpha_(alpha), beta_(beta),
			output_width_(output_width(glp)),
			local_mean_(output_dim(glp)*batch_size, context),
			local_variance_(output_dim(glp)*batch_size, context),
			input_(input_dim(glp)*batch_size, context),
			output_(output_dim(glp)*batch_size, context),
			local_mean_kernel_(make_kernel(local_mean_kernel_source,
				"local_mean", context)),
			local_variance_kernel_(make_kernel(local_variance_kernel_source,
				"local_variance", context)),
			forward_kernel_(make_kernel(
				local_contrast_normalization_forward_kernel_source,
				"forward", context)),
			backward_kernel_(make_kernel(
				local_contrast_normalization_backward_kernel_source,
				"backward", context)) {}

			decltype(auto) batch_size() const { return batch_size_; }

			decltype(auto) input_rank() const { return 3; }
			decltype(auto) output_rank() const { return 3; }

			decltype(auto) input_size(rank_id ri) const {
				return ri == rank_id::width || ri == rank_id::height ? glp_.input_width :
					ri == rank_id::channel_num ? glp_.input_channel_num : 0;
			}
			decltype(auto) output_size(rank_id ri) const {
				return ri == rank_id::width || ri == rank_id::height ? output_width_ :
					ri == rank_id::channel_num ? glp_.output_channel_num : 0;
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(int test_batch_size,
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input) ==
					neu::layer::input_dim(*this)*test_batch_size);
				NEU_ASSERT(neu::range::distance(output) ==
					neu::layer::output_dim(*this)*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				/*
				neu::enqueue_nd_range_kernel<3>(queue, forward_kernel_,
					{0, 0, 0}, {output_width_, output_width_, test_batch_size},
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.output_channel_num),
					static_cast<int>(glp_.stride),
					static_cast<int>(glp_.pad),
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<int>(neu::range::get_begin_index(output)),
					filters_);
				*/

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				range::copy(input, input_, queue);

				// local_mean
				neu::enqueue_nd_range_kernel<3>(queue, local_mean_kernel_,
					{0, 0, 0}, {output_width_, output_width_, batch_size_},
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.stride),
					static_cast<int>(glp_.pad),
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(local_mean_),
					static_cast<int>(neu::range::get_begin_index(local_mean_)));

				// local_variance
				neu::enqueue_nd_range_kernel<3>(queue, local_variance_kernel_,
					{0, 0, 0}, {output_width_, output_width_, batch_size_},
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.stride),
					static_cast<int>(glp_.pad),
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(local_variance_),
					static_cast<int>(neu::range::get_begin_index(local_variance_)),
					local_mean_);

				// forward
				neu::enqueue_nd_range_kernel<1>(queue, forward_kernel_,
					{0}, {layer::whole_output_size(*this)},
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<int>(neu::range::get_begin_index(output)),
					static_cast<float>(alpha_),
					static_cast<float>(beta_),
					local_mean_,
					local_variance_);

				range::copy(output, output_, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				/* do nithing */
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));

				neu::enqueue_nd_range_kernel<1>(queue, backward_kernel_,
					{0}, {layer::whole_input_size(*this)},
					neu::range::get_buffer(prev_delta),
					static_cast<int>(neu::range::get_begin_index(prev_delta)),
					neu::range::get_buffer(delta),
					static_cast<int>(neu::range::get_begin_index(delta)),
					static_cast<int>(glp_.filter_width),
					static_cast<float>(alpha_),
					static_cast<float>(beta_),
					local_mean_,
					local_variance_,
					input_,
					output_);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				/* do nothing */
			}

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "local_contrast_normalization";
				layer::serialize(glp_, emitter);
				emitter 
					<< YAML::Key << "output_width(calculated)"
						<< YAML::Value << output_width_
					<< YAML::Key << "batch_size"
						<< YAML::Value << batch_size_
				<< YAML::EndMap;
			}

		private:
			geometric_layer_property glp_;
			int batch_size_;
			scalar alpha_;
			scalar beta_;

			int output_width_;

			gpu_vector local_mean_;
			gpu_vector local_variance_;
			gpu_vector input_;
			gpu_vector output_;

			kernel local_mean_kernel_;
			kernel local_variance_kernel_;
			kernel forward_kernel_;
			kernel backward_kernel_;
		};
	}
}// namespace neu

#endif //NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_IMPL_HPP
