#ifndef NEU_LAYER_MAX_POOLING_IMPL_HPP
#define NEU_LAYER_MAX_POOLING_IMPL_HPP
//20151026
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/max_pooling/kernel_source.hpp>

namespace neu {
	namespace layer {
		class max_pooling {
		public:
			max_pooling() = default;

			max_pooling(
				geometric_layer_property const& glp,
				int batch_size,
				boost::compute::context const& context)
			: glp_(glp), batch_size_(batch_size),
			pooling_kernel_(make_kernel(max_pooling_kernel_source,
				"max_pooling", context)),
			output_width_(output_width(glp)),
			gpu_indices_(output_dim(glp)*batch_size_, context) {}

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
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<int>(neu::range::get_begin_index(output)),
					gpu_indices_,
					static_cast<int>(glp_.stride),
					static_cast<int>(glp_.input_channel_num),
					static_cast<int>(glp_.input_width), static_cast<int>(output_width_),
					static_cast<int>(glp_.filter_width)); //TODO pad

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
				NEU_ASSERT(neu::range::distance(delta)
					== neu::layer::whole_output_size(*this));
				NEU_ASSERT(neu::range::distance(prev_delta)
					== neu::layer::whole_input_size(*this));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));

				const auto cpu_indices = to_cpu_indices(gpu_indices_, queue);

				const auto cpu_delta = [&delta, &queue](){
					cpu_vector cpu_delta(neu::range::distance(delta));
					neu::range::copy(delta, cpu_delta, queue);
					return cpu_delta;
				}();

				cpu_vector cpu_prev_delta(neu::range::distance(prev_delta), 0.f);
				for(auto i = 0u; i < cpu_indices.size(); ++i) {
					NEU_ASSERT(i < cpu_indices.size());
					NEU_ASSERT(i < cpu_delta.size());
					NEU_ASSERT(static_cast<int>(
						cpu_indices[i]) < cpu_prev_delta.size());

					cpu_prev_delta[cpu_indices[i]] += cpu_delta[i];
				}
				neu::range::copy(cpu_prev_delta, prev_delta, queue);

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue&) { /* do nothing */ }

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "max_pooling";
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

			kernel pooling_kernel_;

			int output_width_;

			gpu_indices gpu_indices_;
		};

		decltype(auto) deserialize_max_pooling(YAML::Node const& node,
				boost::compute::context const& context) {
			const auto glp = deserialize_geometric_layer_property(node);
			return max_pooling(
				glp, node["batch_size"].as<int>(), context);
		}
	}
}// namespace neu

#endif //NEU_LAYER_MAX_POOLING_IMPL_HPP
