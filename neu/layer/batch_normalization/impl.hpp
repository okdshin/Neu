#ifndef NEU_LAYER_BATCH_NORMALIZATION_IMPL_HPP
#define NEU_LAYER_BATCH_NORMALIZATION_IMPL_HPP
//20160215
#include <limits>
#include <yaml-cpp/yaml.h>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/traits.hpp>
#include <neu/layer/batch_normalization/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>
namespace neu {
	namespace layer {
		class batch_normalization {
		public:
			batch_normalization() = default;
			batch_normalization(batch_normalization const&) = default;
			batch_normalization& operator=(batch_normalization const&) = default;
			batch_normalization(batch_normalization&&) = default;
			batch_normalization& operator=(batch_normalization&&) = default;
			~batch_normalization() = default;

			batch_normalization(
				int batch_size,
				int input_dim,
				cpu_vector const& gamma,
				cpu_vector const& beta,
				optimizer::any_optimizer const& gamma_optimizer,
				optimizer::any_optimizer const& beta_optimizer,
				boost::compute::command_queue& queue) 
			: batch_size_(batch_size),
			input_dim_(input_dim),
			output_dim_(input_dim),
			gamma_optimizer_(gamma_optimizer),
			beta_optimizer_(beta_optimizer),
			mean_kernel_(make_kernel(
				batch_normalization_kernel_source, "mean", queue.get_context())),
			variance_kernel_(make_kernel(
				batch_normalization_kernel_source, "variance", queue.get_context())),
			normalize_kernel_(make_kernel(
				batch_normalization_kernel_source, "normalize_input", queue.get_context())),
			forward_kernel_(make_kernel(
				batch_normalization_kernel_source, "forward", queue.get_context())),
			del_normalized_input_kernel_(make_kernel(
				batch_normalization_kernel_source, "del_normalized_input", queue.get_context())),
			del_variance_kernel_(make_kernel(
				batch_normalization_kernel_source, "del_variance", queue.get_context())),
			del_mean_kernel_(make_kernel(
				batch_normalization_kernel_source, "del_mean", queue.get_context())),
			backward_kernel_(make_kernel(
				batch_normalization_kernel_source, "backward", queue.get_context())),
			del_gamma_kernel_(make_kernel(
				batch_normalization_kernel_source, "del_gamma", queue.get_context())),
			del_beta_kernel_(make_kernel(
				batch_normalization_kernel_source, "del_beta", queue.get_context())),

			input_(input_dim*batch_size,
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			delta_(input_dim*batch_size,
				std::numeric_limits<cl_float>::quiet_NaN(), queue),

			mean_(input_dim, std::numeric_limits<cl_float>::quiet_NaN(), queue),
			variance_(input_dim, std::numeric_limits<cl_float>::quiet_NaN(), queue),
			normalized_input_(input_dim*batch_size, queue.get_context()),
			gamma_(gamma.begin(), gamma.end(), queue),
			beta_(beta.begin(), beta.end(), queue),
			
			del_mean_(input_dim, std::numeric_limits<cl_float>::quiet_NaN(), queue),
			del_variance_(input_dim, std::numeric_limits<cl_float>::quiet_NaN(), queue),
			del_normalized_input_(input_dim*batch_size,
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			del_gamma_(gamma_.size(), std::numeric_limits<cl_float>::quiet_NaN(), queue),
			del_beta_(beta_.size(), std::numeric_limits<cl_float>::quiet_NaN(), queue)
			{
				if(range::distance(gamma_) != input_dim) {
					throw std::invalid_argument("the size of gamma is not correct.");
				}
				if(range::distance(beta_) != input_dim) {
					throw std::invalid_argument("the size of beta is not correct.");
				}
			}

			decltype(auto) input_rank() const { return 1; }
			decltype(auto) output_rank() const { return 1; }
			decltype(auto) input_size(rank_id ri) const {
				return ri == rank_id::dim ? input_dim_ : 0; }
			decltype(auto) output_size(rank_id ri) const {
				return ri == rank_id::dim ? output_dim_ : 0; }
			decltype(auto) batch_size() const { return batch_size_; }
			decltype(auto) mean(boost::compute::command_queue& queue) const {
				return to_cpu_vector(mean_, queue); }
			decltype(auto) variance(boost::compute::command_queue& queue) const {
				return to_cpu_vector(variance_, queue); }
			decltype(auto) normalized_input(boost::compute::command_queue& queue) const {
				return to_cpu_vector(normalized_input_, queue); }
			decltype(auto) del_gamma(boost::compute::command_queue& queue) const {
				return to_cpu_vector(del_gamma_, queue); }
			decltype(auto) del_beta(boost::compute::command_queue& queue) const {
				return to_cpu_vector(del_beta_, queue); }
			decltype(auto) gamma(boost::compute::command_queue& queue) const {
				return to_cpu_vector(gamma_, queue); }
			decltype(auto) beta(boost::compute::command_queue& queue) const {
				return to_cpu_vector(beta_, queue); }
			decltype(auto) gamma_optimizer() const { return (gamma_optimizer_); }
			decltype(auto) beta_optimizer() const { return (beta_optimizer_); }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(int test_batch_size,
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == input_dim_*test_batch_size);
				NEU_ASSERT(range::distance(output) ==  output_dim_*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				assert(!"batch_normalization test_forward is not implemented!");
				/* //TODO total_mean, total_variance
				enqueue_nd_range_kernel<2>(queue, test_forward_kernel_,
					{0, 0}, {output_dim_, test_batch_size},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					range::get_buffer(output),
					static_cast<cl_int>(range::get_begin_index(output)),
					gamma_, beta_, total_variance_, total_mean_,
					static_cast<cl_int>(input_dim_));
				*/
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == range::distance(input_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				range::copy(input, input_, queue);
				enqueue_nd_range_kernel<1>(queue, mean_kernel_,
					{0}, {output_dim_},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					mean_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				enqueue_nd_range_kernel<1>(queue, variance_kernel_,
					{0}, {output_dim_},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					variance_,
					mean_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				enqueue_nd_range_kernel<2>(queue, normalize_kernel_,
					{0, 0}, {output_dim_, batch_size_},
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					normalized_input_,
					mean_,
					variance_,
					eta_, //TODO eta
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				enqueue_nd_range_kernel<2>(queue, forward_kernel_,
					{0, 0}, {output_dim_, batch_size_},
					normalized_input_,
					range::get_buffer(output),
					static_cast<cl_int>(range::get_begin_index(output)),
					gamma_,
					beta_,
					static_cast<cl_int>(input_dim_));
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == range::distance(delta_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				range::copy(delta, delta_, queue); //TODO async operation
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == range::distance(delta_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(gamma_, queue));

				backward_top(delta, queue);

				// del_normalized_input
				enqueue_nd_range_kernel<2>(queue, del_normalized_input_kernel_,
					{0, 0}, {input_dim_, batch_size_},
					range::get_buffer(delta),
					static_cast<cl_int>(range::get_begin_index(delta)),
					gamma_,
					del_normalized_input_,
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(
					is_all_of_finite(del_normalized_input_, queue));

				// del_variance
				enqueue_nd_range_kernel<1>(queue, del_variance_kernel_,
					{0}, {input_dim_},
					range::get_buffer(input_),
					static_cast<cl_int>(range::get_begin_index(input_)),
					del_normalized_input_, mean_, variance_, eta_,
					del_variance_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_variance_, queue));
				
				// del_mean
				enqueue_nd_range_kernel<1>(queue, del_mean_kernel_,
					{0}, {input_dim_},
					range::get_buffer(input_),
					static_cast<cl_int>(range::get_begin_index(input_)),
					del_normalized_input_, mean_, variance_, del_variance_, eta_,
					del_mean_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_mean_, queue));

				// prev_delta
				enqueue_nd_range_kernel<2>(queue, backward_kernel_,
					{0, 0}, {output_dim_, batch_size_},
					range::get_buffer(input_),
					static_cast<cl_int>(range::get_begin_index(input_)),
					del_normalized_input_, mean_, variance_, del_variance_, del_mean_,
					eta_,
					range::get_buffer(prev_delta),
					static_cast<cl_int>(range::get_begin_index(prev_delta)),
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				// del_gamma
				enqueue_nd_range_kernel<1>(queue, del_gamma_kernel_,
					{0}, {output_dim_},
					normalized_input_, delta_, del_gamma_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_gamma_, queue));
				gamma_optimizer_.apply(gamma_, del_gamma_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(gamma_, queue));

				// del_beta
				enqueue_nd_range_kernel<1>(queue, del_beta_kernel_,
					{0}, {output_dim_},
					delta_, del_beta_,
					static_cast<cl_int>(batch_size_),
					static_cast<cl_int>(input_dim_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_beta_, queue));
				beta_optimizer_.apply(beta_, del_beta_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(beta_, queue));
			}

		private:
			int batch_size_;
			int input_dim_;
			int output_dim_;

			optimizer::any_optimizer gamma_optimizer_, beta_optimizer_;
			kernel mean_kernel_, variance_kernel_, normalize_kernel_, forward_kernel_,
				del_normalized_input_kernel_, del_variance_kernel_, del_mean_kernel_,
				backward_kernel_, del_gamma_kernel_, del_beta_kernel_;
			gpu_vector input_, delta_;

			gpu_vector mean_;
			gpu_vector variance_;
			gpu_vector normalized_input_;
			gpu_vector gamma_;
			gpu_vector beta_;

			gpu_vector del_mean_;
			gpu_vector del_variance_;
			gpu_vector del_normalized_input_;
			gpu_vector del_gamma_;
			gpu_vector del_beta_;

			const scalar eta_ = 1.e-4f;
		};
		decltype(auto) make_batch_normalization(
				int batch_size,
				int input_dim,
				optimizer::any_optimizer const& gamma_optimizer,
				optimizer::any_optimizer const& beta_optimizer,
				boost::compute::command_queue& queue) {
			return neu::layer::batch_normalization(batch_size, input_dim,
				cpu_vector(input_dim, 1.f),
				cpu_vector(input_dim, 0.f),
				gamma_optimizer,
				beta_optimizer,
				queue);
		}

		namespace traits {
			template<>
			class serialize<batch_normalization> {
			public:
				static decltype(auto) call(batch_normalization const& bn,
						YAML::Emitter& emitter, boost::compute::command_queue& queue) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "batch_normalization"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(bn)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(bn)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(bn)
						<< YAML::Key << "gamma"
							<< YAML::Value << YAML::Flow << bn.gamma(queue)
						<< YAML::Key << "beta"
							<< YAML::Value << YAML::Flow << bn.beta(queue)
						<< YAML::Key << "gamma_optimizer"
							<< YAML::Value;
					optimizer::serialize(bn.gamma_optimizer(), emitter, queue);
					emitter << YAML::Key << "beta_optimizer"
							<< YAML::Value;
					optimizer::serialize(bn.beta_optimizer(), emitter, queue);
					emitter << YAML::EndMap;
				}
			};
		}

		decltype(auto) deserialize_batch_normalization(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			return batch_normalization(
				node["batch_size"].as<int>(),
				node["input_dim"].as<int>(),
				node["gamma"].as<cpu_vector>(),
				node["beta"].as<cpu_vector>(),
				optimizer::deserialize(node["gamma_optimizer"], queue),
				optimizer::deserialize(node["beta_optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_BATCH_NORMALIZATION_LAYER_IMPL_HPP
