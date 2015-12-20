#ifndef NEU_LAYER_INNER_PRODUCT_IMPL_HPP
#define NEU_LAYER_INNER_PRODUCT_IMPL_HPP
//20151023
#include <yaml-cpp/yaml.h>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/inner_product/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>
namespace neu {
	namespace layer {
		class inner_product {
		public:
			inner_product() = default;
			inner_product(inner_product const&) = default;
			inner_product& operator=(inner_product const&) = default;
			inner_product(inner_product&&) = default;
			inner_product& operator=(inner_product&&) = default;
			~inner_product() = default;

			inner_product(
				std::size_t input_dim,
				std::size_t output_dim,
				std::size_t batch_size,
				cpu_vector const& weight,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) 
			: input_dim_(input_dim),
			output_dim_(output_dim),
			batch_size_(batch_size),
			weight_(weight.begin(), weight.end(), queue),
			optimizer_(optimizer),
			forward_kernel_(make_kernel(
				inner_product_forward_kernel_source, "forward", queue.get_context())),
			backward_kernel_(make_kernel(
				inner_product_backward_kernel_source, "backward", queue.get_context())),
			update_kernel_(make_kernel(
				inner_product_update_kernel_source, "update", queue.get_context())),
			input_(input_dim*batch_size, queue.get_context()),
			delta_(output_dim*batch_size, queue.get_context()),
			del_weight_(weight_.size(), queue.get_context()) {
				if(weight_.size() != input_dim*output_dim) {
					throw std::invalid_argument("the size of weight is not correct.");
				}
			}

			decltype(auto) input_rank() const { return 1; }
			decltype(auto) output_rank() const { return 1; }
			decltype(auto) input_size(rank_id ri) const {
				return static_cast<std::size_t>(ri) == 1 ? input_dim_ : 0; }
			decltype(auto) output_size(rank_id ri) const {
				return static_cast<std::size_t>(ri) == 1 ? output_dim_ : 0; }
			decltype(auto) batch_size() const { return batch_size_; }
			decltype(auto) weight(boost::compute::command_queue& queue) const {
				return to_cpu_vector(weight_, queue); }
			decltype(auto) optimizer() const { return (optimizer_); }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(std::size_t test_batch_size,
					InputRange const& input, OutputRange const& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == input_dim_*test_batch_size);
				NEU_ASSERT(range::distance(output) ==  output_dim_*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				enqueue_nd_range_kernel<2>(queue, forward_kernel_,
					{0, 0}, {output_dim_, test_batch_size},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					range::get_buffer(output),
					static_cast<cl_int>(range::get_begin_index(output)),
					weight_,
					static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange const& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == input_.size());
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				range::copy(input, input_, queue); //TODO async operation
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
					InputRange const& delta, OutputRange const& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == delta_.size());
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				backward_top(delta, queue);
				enqueue_nd_range_kernel<2>(queue, backward_kernel_,
					{0, 0}, {input_dim_, batch_size_},
					range::get_buffer(prev_delta),
					static_cast<cl_int>(range::get_begin_index(prev_delta)),
					range::get_buffer(delta),
					static_cast<cl_int>(range::get_begin_index(delta)),
					weight_,
					static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				enqueue_nd_range_kernel<2>(queue, update_kernel_,
					{0, 0}, {input_dim_, output_dim_},
					input_, delta_, del_weight_,
					static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_),
					static_cast<cl_int>(batch_size_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_weight_, queue));
				optimizer_.apply(weight_, del_weight_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(weight_, queue));
			}

		private:
			std::size_t input_dim_;
			std::size_t output_dim_;
			std::size_t batch_size_;
			gpu_vector weight_;

			optimizer::any_optimizer optimizer_;
			kernel forward_kernel_, backward_kernel_, update_kernel_;
			gpu_vector input_, delta_;
			gpu_vector del_weight_;
		};
		template<typename WeightGen>
		decltype(auto) make_inner_product(
				std::size_t input_dim,
				std::size_t output_dim,
				std::size_t batch_size,
				WeightGen const& wg,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) {
			cpu_vector cpu_weight(input_dim*output_dim);
			std::generate(cpu_weight.begin(), cpu_weight.end(), wg);
			return neu::layer::inner_product(input_dim, output_dim, batch_size,
				cpu_weight, optimizer, queue);
		}

		namespace traits {
			template<>
			class serialize<inner_product> {
			public:
				static decltype(auto) call(inner_product const& ip,
						YAML::Emitter& emitter, boost::compute::command_queue& queue) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "inner_product"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(ip)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(ip)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(ip)
						<< YAML::Key << "weight"
							<< YAML::Value << YAML::Flow << ip.weight(queue)
						<< YAML::Key << "optimizer"
							<< YAML::Value;
					optimizer::serialize(ip.optimizer(), emitter, queue);
					emitter << YAML::EndMap;
				}
			};
		}

		decltype(auto) deserialize_inner_product(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			return inner_product(
				node["input_dim"].as<std::size_t>(),
				node["output_dim"].as<std::size_t>(),
				node["batch_size"].as<std::size_t>(),
				node["weight"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_INNER_PRODUCT_LAYER_IMPL_HPP
