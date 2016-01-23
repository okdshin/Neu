#ifndef NEU_LAYER_CONVOLUTION_IMPL_HPP
#define NEU_LAYER_CONVOLUTION_IMPL_HPP
//20151005
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/impl/convolution_indices.hpp>
#include <neu/layer/convolution/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>

namespace neu {
	namespace layer {
		class convolution {
		public:
			convolution() = default;

			convolution(
				geometric_layer_property const& glp,
				int batch_size,
				cpu_vector const& filters,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue)
			: glp_(glp), batch_size_(batch_size),
			filters_(filters.begin(), filters.end(), queue),
			optimizer_(optimizer),
			output_width_(output_width(glp)),
			indices_(impl::make_convolution_indices(
				glp_.input_width, output_width_, glp_.filter_width,
				glp_.stride, glp_.pad, queue)),
			convolution_kernel_(make_kernel(convolution_kernel_source,
				"convolution", queue.get_context())),
			convolution_back_kernel_(make_kernel(convolution_back_kernel_source,
				"convolution_back", queue.get_context())),
			update_del_filters_kernel_(make_kernel(update_delta_filters_kernel_source, 
				"update_delta_filters", queue.get_context())),
			input_(layer::input_dim(glp)*batch_size_, queue.get_context()),
			delta_(layer::output_dim(glp)*batch_size_, queue.get_context()),
			del_filters_(filters_.size(), queue.get_context()) {}

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

			decltype(auto) filters(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(filters_, queue);
			}
			decltype(auto) del_filters(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(del_filters_, queue);
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

				neu::enqueue_nd_range_kernel<3>(queue, convolution_kernel_,
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

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input)
					== static_cast<int>(input_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				neu::range::copy(input, input_, queue); //TODO async operation
				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& next_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(next_delta)
					== static_cast<int>(delta_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(next_delta, queue));

				neu::range::copy(next_delta, delta_, queue); //TODO async operation
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& next_delta, OutputRange& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(next_delta)
					== static_cast<int>(delta_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(next_delta, queue));

				backward_top(next_delta, queue);
				neu::enqueue_nd_range_kernel<2>(queue, convolution_back_kernel_,
					{0, 0}, {glp_.input_width*glp_.input_width, batch_size_},
					indices_.indices_range_list_for_input,
					indices_.output_indices_list_for_input,
					indices_.filter_indices_list_for_input,
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.output_channel_num),
					neu::range::get_buffer(delta),
					static_cast<int>(neu::range::get_begin_index(delta)),
					neu::range::get_buffer(next_delta),
					static_cast<int>(neu::range::get_begin_index(next_delta)),
					filters_);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				neu::enqueue_nd_range_kernel<3>(queue, update_del_filters_kernel_,
					{0, 0, 0},
					{glp_.filter_width, glp_.filter_width, glp_.output_channel_num},
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.output_channel_num),
					static_cast<int>(glp_.stride),
					static_cast<int>(glp_.pad),
					static_cast<int>(batch_size_),
					input_, delta_, del_filters_);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_filters_, queue));

				optimizer_.apply(filters_, del_filters_, queue);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(filters_, queue));
			}

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "convolution";
				layer::serialize(glp_, emitter);
				emitter 
					<< YAML::Key << "output_width(calculated)"
						<< YAML::Value << output_width_
					<< YAML::Key << "batch_size"
						<< YAML::Value << batch_size_
					<< YAML::Key << "filters"
						<< YAML::Value << YAML::Flow << filters(queue)
					<< YAML::Key << "optimizer"
						<< YAML::Value;
				optimizer::serialize(optimizer_, emitter, queue);
				emitter << YAML::EndMap;
			}

		private:
			geometric_layer_property glp_;
			int batch_size_;

			gpu_vector filters_;

			optimizer::any_optimizer optimizer_;

			int output_width_;

			impl::convolution_indices indices_;

			kernel convolution_kernel_;
			kernel convolution_back_kernel_;
			kernel update_del_filters_kernel_;

			gpu_vector input_;
			gpu_vector delta_;

			gpu_vector del_filters_;
		};

		template<typename FilterGen>
		decltype(auto) make_convolution(
			geometric_layer_property const& glp,
			int batch_size,
			FilterGen const& fg,
			optimizer::any_optimizer const& optimizer,
			boost::compute::command_queue& queue
		) {
			cpu_vector cpu_filters(filters_size(glp));
			std::generate(cpu_filters.begin(), cpu_filters.end(), fg);
			return convolution(glp, batch_size, cpu_filters, optimizer, queue);
		}

		decltype(auto) deserialize_convolution(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto glp = deserialize_geometric_layer_property(node);
			return convolution(
				glp, node["batch_size"].as<int>(),
				node["filters"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_CONVOLUTION_IMPL_HPP
