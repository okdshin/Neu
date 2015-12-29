#ifndef NEU_BIAS_LAYER_IMPL_HPP
#define NEU_BIAS_LAYER_IMPL_HPP
//20151023
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/bias/kernel_source.hpp>
#include <neu/optimizer/any_optimizer.hpp>
namespace neu {
	namespace layer {
		class bias {
		public:
			bias() = default;
			bias(bias const&) = default;
			bias& operator=(bias const&) = default;
			bias(bias&&) = default;
			bias& operator=(bias&&) = default;
			~bias() = default;

			bias(
				int input_dim,
				int batch_size,
				cpu_vector const& bias,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue)
			: input_dim_(input_dim),
			batch_size_(batch_size),
			bias_(bias.begin(), bias.end(), queue),
			optimizer_(optimizer),
			forward_kernel_(make_kernel(bias_forward_kernel_source,
				"forward", queue.get_context())),
			update_kernel_(make_kernel(bias_update_kernel_source,
				"update", queue.get_context())),
			delta_(input_dim*batch_size, queue.get_context()),
			del_bias_(bias_.size(), queue.get_context()) {
				if(bias_.size() != input_dim) {
					throw std::invalid_argument("the size of bias is not correct.");
				}	
			}

			decltype(auto) input_rank() const { return 1; }
			decltype(auto) output_rank() const { return 1; }
			decltype(auto) input_size(rank_id ri) const { 
				return ri == rank_id::dim ? input_dim_ : 0; }
			decltype(auto) output_size(rank_id ri) const { return input_size(ri); }
			decltype(auto) batch_size() const { return batch_size_; }
			decltype(auto) weight(boost::compute::command_queue& queue) const {
				return to_cpu_vector(bias_, queue); }
			decltype(auto) optimizer() const { return (optimizer_); }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(int test_batch_size,
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == input_dim_*test_batch_size);
				NEU_ASSERT(range::distance(output) ==  input_dim_*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				enqueue_nd_range_kernel<2>(queue, forward_kernel_,
					{0, 0}, {input_dim_, test_batch_size},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					range::get_buffer(output),
					static_cast<cl_int>(range::get_begin_index(output)),
					bias_,
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == delta_.size());
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				range::copy(delta, delta_, queue); //TODO async operation
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == delta_.size());
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				backward_top(delta, queue);
				range::copy(delta, prev_delta, queue); //TODO async operation
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				enqueue_nd_range_kernel<1>(queue, update_kernel_,
					{0}, {input_dim_}, delta_, del_bias_,
					static_cast<cl_int>(input_dim_), static_cast<cl_int>(batch_size_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_bias_, queue));
				optimizer_.apply(bias_, del_bias_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(bias_, queue));
			}

		private:
			int input_dim_;
			int batch_size_;
			gpu_vector bias_;
			optimizer::any_optimizer optimizer_;

			kernel forward_kernel_, update_kernel_;
			gpu_vector delta_;
			gpu_vector del_bias_;
		};
		template<typename BiasGen>
		decltype(auto) make_bias(
				int input_dim,
				int batch_size,
				BiasGen const& bg,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) {
			cpu_vector cpu_bias(input_dim);
			std::generate(cpu_bias.begin(), cpu_bias.end(), bg);
			return neu::layer::bias(input_dim, batch_size, cpu_bias, optimizer, queue);
		}

		namespace traits {
			template<>
			class serialize<bias> {
			public:
				static decltype(auto) call(
						bias const& b,
						YAML::Emitter& emitter, boost::compute::command_queue& queue) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "bias"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(b)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(b)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(b)
						<< YAML::Key << "weight"
							<< YAML::Value << YAML::Flow << b.weight(queue)
						<< YAML::Key << "optimizer"
							<< YAML::Value;
					optimizer::serialize(b.optimizer(), emitter, queue);
					emitter << YAML::EndMap;
				}
			};
		}

		decltype(auto) deserialize_bias(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			return bias(
				node["input_dim"].as<int>(),
				node["batch_size"].as<int>(),
				node["weight"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_BIAS_LAYER_IMPL_HPP
