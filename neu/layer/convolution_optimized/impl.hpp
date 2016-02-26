#ifndef NEU_LAYER_CONVOLUTION_OPTIMIZED_IMPL_HPP
#define NEU_LAYER_CONVOLUTION_OPTIMIZED_IMPL_HPP
//20151005
#include <random>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/convolution_optimized/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>

namespace neu {
	namespace layer {
		class convolution_optimized {
		public:
			convolution_optimized() = default;

			convolution_optimized(
				geometric_layer_property const& glp,
				int batch_size,
				cpu_vector const& filters,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue)
			: glp_(glp), batch_size_(batch_size),
			filters_(filters.begin(), filters.end(), queue),
			optimizer_(optimizer),
			output_width_(output_width(glp)),
			reorder_input_kernel_(make_kernel(convolution_optimized_kernel_source,
				"reorder_input", queue.get_context())),
			forward_kernel_(make_kernel(convolution_optimized_kernel_source,
				"forward", queue.get_context())),
			reorder_filters_kernel_(make_kernel(convolution_optimized_kernel_source,
				"reorder_filters", queue.get_context())),
			reorder_delta_kernel_(make_kernel(convolution_optimized_kernel_source,
				"reorder_delta", queue.get_context())),
			backward_kernel_(make_kernel(convolution_optimized_kernel_source,
				"backward", queue.get_context())),
			update_kernel_(make_kernel(convolution_optimized_kernel_source,
				"update", queue.get_context())),
			reordered_input_(
				glp.filter_width*glp.filter_width*glp.input_channel_num*
				output_width_*output_width_*batch_size_,
				std::numeric_limits<cl_float>::quiet_NaN(),
				queue),
			delta_(layer::output_dim(glp)*batch_size_, queue.get_context()),
			reordered_filters_(filters_.size(),
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			reordered_delta_(
				glp.filter_width*glp.filter_width*glp.output_channel_num*
				glp_.input_width*glp_.input_width*batch_size_,
				/*std::numeric_limits<cl_float>::quiet_NaN()*/0.f, queue),
			del_filters_(filters_.size(),
				std::numeric_limits<cl_float>::quiet_NaN(), queue) {}

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

			decltype(auto) reordered_input(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(reordered_input_, queue);
			}
			decltype(auto) filters(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(filters_, queue);
			}
			decltype(auto) reordered_delta(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(reordered_delta_, queue);
			}
			decltype(auto) reordered_filters(boost::compute::command_queue& queue) const {
				return neu::to_cpu_vector(reordered_filters_, queue);
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

				//reorder input
				neu::enqueue_nd_range_kernel<3>(queue, reorder_input_kernel_,
					{0, 0, 0}, {output_width_, output_width_, test_batch_size},
					neu::range::get_buffer(input),
					static_cast<cl_int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(reordered_input_),
					static_cast<cl_int>(glp_.input_width),
					static_cast<cl_int>(output_width_),
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.output_channel_num),
					static_cast<cl_int>(glp_.stride),
					static_cast<cl_int>(glp_.pad));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(
					is_all_of_finite(reordered_input_, queue));

				// forward
				neu::enqueue_nd_range_kernel<3>(queue, forward_kernel_,
					{0, 0, 0},
					{output_width_*output_width_, glp_.output_channel_num,
						test_batch_size},
					reordered_input_,
					filters_,
					neu::range::get_buffer(output),
					static_cast<cl_int>(neu::range::get_begin_index(output)),
					static_cast<cl_int>(output_width_),
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.output_channel_num));

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input)
					== neu::layer::input_dim(glp_)*batch_size_);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(delta)
					== static_cast<int>(delta_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				neu::range::copy(delta, delta_, queue); //TODO async operation
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				// reorder filters
				NEU_ASSERT(neu::range::distance(delta)
					== static_cast<int>(delta_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				backward_top(delta, queue);
				neu::enqueue_nd_range_kernel<3>(queue,
					reorder_filters_kernel_,
					{0,0,0},
					{glp_.filter_width*glp_.filter_width,
						glp_.input_channel_num, glp_.output_channel_num},
					filters_,
					reordered_filters_,
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.output_channel_num));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(
					is_all_of_finite(reordered_filters_, queue));

				// reorder delta
				neu::enqueue_nd_range_kernel<3>(queue,
					reorder_delta_kernel_,
					{0,0,0}, {output_width_, output_width_, batch_size_},
					neu::range::get_buffer(delta),
					static_cast<cl_int>(neu::range::get_begin_index(delta)),
					reordered_delta_,
					static_cast<cl_int>(glp_.input_width),
					static_cast<cl_int>(output_width_),
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.output_channel_num),
					static_cast<cl_int>(glp_.stride),
					static_cast<cl_int>(glp_.pad));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(
					is_all_of_finite(reordered_delta_, queue));

				// backward
				neu::enqueue_nd_range_kernel<3>(queue,
					backward_kernel_,
					{0,0,0},
					{glp_.input_width*glp_.input_width,
						glp_.input_channel_num, batch_size_},
					reordered_delta_,
					reordered_filters_,
					neu::range::get_buffer(prev_delta),
					static_cast<cl_int>(neu::range::get_begin_index(prev_delta)),
					static_cast<cl_int>(glp_.input_width),
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.output_channel_num));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				const int ffk_size =
					glp_.filter_width*glp_.filter_width*glp_.input_channel_num;
				const int oo_size = output_width_*output_width_;
				neu::enqueue_nd_range_kernel<2>(queue, update_kernel_,
					{0, 0},
					{ffk_size, glp_.output_channel_num},
					reordered_input_, delta_, del_filters_,
					static_cast<int>(ffk_size),
					static_cast<int>(oo_size),
					static_cast<int>(glp_.output_channel_num),
					static_cast<int>(batch_size_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_filters_, queue));

				optimizer_.apply(filters_, del_filters_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(filters_, queue));
			}

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "convolution_optimized";
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

			kernel reorder_input_kernel_, forward_kernel_,
				reorder_filters_kernel_, reorder_delta_kernel_, backward_kernel_,
				update_kernel_;

			gpu_vector reordered_input_;
			
			gpu_vector delta_;
			gpu_vector reordered_filters_;
			gpu_vector reordered_delta_;

			gpu_vector del_filters_;
		};

		template<typename FilterGen>
		decltype(auto) make_convolution_optimized(
			geometric_layer_property const& glp,
			int batch_size,
			FilterGen const& fg,
			optimizer::any_optimizer const& optimizer,
			boost::compute::command_queue& queue
		) {
			cpu_vector cpu_filters(neu::layer::filters_size(glp));
			std::generate(cpu_filters.begin(), cpu_filters.end(), fg);
			return convolution_optimized(glp, batch_size, cpu_filters, optimizer, queue);
		}

		template<typename Rng>
		decltype(auto) make_convolution_optimized_xavier(
			geometric_layer_property const& glp,
			int batch_size,
			Rng&& rng,
			optimizer::any_optimizer const& optimizer,
			boost::compute::command_queue& queue
		) {
			auto dist = std::normal_distribution<neu::scalar>(
				0.f, std::sqrt(1.f/neu::layer::input_dim(glp)));
			return neu::layer::make_convolution_optimized(glp, batch_size,
				[&rng, dist]() mutable { return dist(rng); },
				optimizer, queue);
		}

		decltype(auto) deserialize_convolution_optimized(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto glp = deserialize_geometric_layer_property(node);
			return convolution_optimized(
				glp, node["batch_size"].as<int>(),
				node["filters"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_CONVOLUTION_OPTIMIZED_IMPL_HPP
