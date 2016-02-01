#ifndef NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
#define NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
//20151026
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/average_pooling/kernel_source.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/impl/convolution_indices.hpp>

namespace neu {
	namespace layer {
		class average_pooling {
		public:
			average_pooling() = default;

			average_pooling(
				geometric_layer_property const& glp,
				int batch_size,
				cpu_vector const& filter,
				boost::compute::command_queue& queue)
			: glp_(glp), batch_size_(batch_size),
			filter_(filter.begin(), filter.end(), queue),
			output_width_(neu::layer::output_width(glp)),
			indices_(impl::make_convolution_indices(
				glp_.input_width, output_width_, glp_.filter_width,
				glp_.stride, glp_.pad, queue)),
			pooling_kernel_(make_kernel(average_pooling_kernel_source,
				"average_pooling", queue.get_context())),
			pooling_back_kernel_(make_kernel(average_pooling_back_kernel_source,
				"average_pooling_back", queue.get_context())) {}

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

				enqueue_nd_range_kernel<3>(queue, pooling_kernel_,
					{0, 0, 0}, {output_width_, output_width_, test_batch_size},
					static_cast<cl_int>(glp_.input_width),
					static_cast<cl_int>(output_width_),
					static_cast<cl_int>(glp_.filter_width),
					static_cast<cl_int>(glp_.input_channel_num),
					static_cast<cl_int>(glp_.stride), static_cast<cl_int>(glp_.pad),
					neu::range::get_buffer(input),
					static_cast<cl_int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<cl_int>(neu::range::get_begin_index(output)),
					filter_);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input) ==
					neu::layer::whole_input_size(*this));
				NEU_ASSERT(neu::range::distance(output) ==
					neu::layer::whole_output_size(*this));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				/* do nothing */
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(delta) ==
					neu::layer::whole_output_size(*this));
				NEU_ASSERT(neu::range::distance(prev_delta) ==
					neu::layer::whole_input_size(*this));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));

				enqueue_nd_range_kernel<3>(queue, pooling_back_kernel_,
					{0, 0, 0},
					{glp_.input_width*glp_.input_width,
						glp_.input_channel_num, batch_size_},
					indices_.indices_range_list_for_input,
					indices_.output_indices_list_for_input,
					indices_.filter_indices_list_for_input,
					static_cast<int>(glp_.input_width),
					static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width),
					static_cast<int>(glp_.input_channel_num),
					neu::range::get_buffer(prev_delta),
					static_cast<int>(neu::range::get_begin_index(prev_delta)),
					neu::range::get_buffer(delta),
					static_cast<int>(neu::range::get_begin_index(delta)),
					filter_);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue&) { /* do nothing */ }

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "average_pooling";
				layer::serialize(glp_, emitter);
				emitter 
					<< YAML::Key << "output_width(calculated)"
						<< YAML::Value << output_width_
					<< YAML::Key << "batch_size"
						<< YAML::Value << batch_size_
					<< YAML::Key << "filter"
						<< YAML::Value << YAML::Flow << neu::to_cpu_vector(filter_, queue)
				<< YAML::EndMap;
			}

		private:
			geometric_layer_property glp_;
			int batch_size_;

			gpu_vector filter_;

			int output_width_;

			impl::convolution_indices indices_;

			kernel pooling_kernel_;
			kernel pooling_back_kernel_;
		};

		decltype(auto) make_average_pooling(
			geometric_layer_property const& glp,
			int batch_size, 
			cpu_vector const& filter,
			boost::compute::command_queue& queue
		){
			return average_pooling(glp, batch_size, filter, queue);
		}

		decltype(auto) make_uniform_average_pooling(
			geometric_layer_property const& glp,
			int batch_size, 
			boost::compute::command_queue& queue
		){
			const auto filter_size = glp.filter_width*glp.filter_width;
			return make_average_pooling(glp, batch_size,
				cpu_vector(filter_size, 1.f/filter_size), queue);
		}

		decltype(auto) deserialize_average_pooling(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto glp = deserialize_geometric_layer_property(node);
			return average_pooling(glp,
				node["batch_size"].as<int>(),
				node["filter"].as<cpu_vector>(),
				queue);
		}
	}
}// namespace neu

#endif //NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
