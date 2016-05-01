#ifndef NEU_LAYER_INNER_PRODUCT_WR_IMPL_HPP
#define NEU_LAYER_INNER_PRODUCT_WR_IMPL_HPP
//20151023
#include <random>
#include <yaml-cpp/yaml.h>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/traits.hpp>
#include <neu/layer/inner_product_wr/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>
#include <neu/layer/impl/matrix_transpose.hpp>
#include <neu/layer/impl/matrix_multiply.hpp>
namespace neu {
	namespace layer {
		class inner_product_wr {
		public:
			inner_product_wr() = default;
			inner_product_wr(inner_product_wr const&) = default;
			inner_product_wr& operator=(inner_product_wr const&) = default;
			inner_product_wr(inner_product_wr&&) = default;
			inner_product_wr& operator=(inner_product_wr&&) = default;
			~inner_product_wr() = default;

			inner_product_wr(
				int input_dim,
				int output_dim,
				int batch_size,
				scalar c,
				cpu_vector const& weight,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) 
			: input_dim_(input_dim),
			output_dim_(output_dim),
			batch_size_(batch_size),
			c_(c),
			weight_(weight.begin(), weight.end(), queue),
			transposed_weight_(weight.size(),
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			squared_weight_(weight.size(),
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			optimizer_(optimizer),
			forward_kernel_(make_kernel(
				inner_product_wr_forward_kernel_source, "forward", queue.get_context())),
			backward_kernel_(make_kernel(
				inner_product_wr_backward_kernel_source, "backward", queue.get_context())),
			update_kernel_(make_kernel(
				inner_product_wr_update_kernel_source, "update", queue.get_context())),
			input_(input_dim*batch_size, queue.get_context()),
			delta_(output_dim*batch_size, queue.get_context()),
			del_weight_(weight_.size(), queue.get_context()) {
				if(range::distance(weight_) != input_dim*output_dim) {
					throw std::invalid_argument("the size of weight is not correct.");
				}
			}

			decltype(auto) input_rank() const { return 1; }
			decltype(auto) output_rank() const { return 1; }
			decltype(auto) input_size(rank_id ri) const {
				return ri == rank_id::dim ? input_dim_ : 0; }
			decltype(auto) output_size(rank_id ri) const {
				return ri == rank_id::dim ? output_dim_ : 0; }
			decltype(auto) batch_size() const { return batch_size_; }
			decltype(auto) c() const { return c_; }
			decltype(auto) del_weight(boost::compute::command_queue& queue) const {
				return to_cpu_vector(del_weight_, queue); }
			decltype(auto) weight(boost::compute::command_queue& queue) const {
				return to_cpu_vector(weight_, queue); }
			decltype(auto) optimizer() const { return (optimizer_); }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(int test_batch_size,
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == input_dim_*test_batch_size);
				NEU_ASSERT(range::distance(output) ==  output_dim_*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				neu::layer::impl::matrix_multiply(input, weight_, output,
					test_batch_size, output_dim_, input_dim_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == range::distance(input_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
				range::copy(input, input_, queue);
				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == range::distance(delta_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				range::copy(delta, delta_, queue);
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(delta) == range::distance(delta_));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				backward_top(delta, queue);
				neu::layer::impl::matrix_transpose(
					weight_, transposed_weight_, input_dim_, output_dim_, queue);
				neu::layer::impl::matrix_multiply(delta, transposed_weight_, prev_delta,
					batch_size_, input_dim_, output_dim_, queue);
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

				// weight regularization
				scalar sum_of_squared = 0.f;
				using boost::compute::lambda::_1;
				boost::compute::transform(weight_.begin(), weight_.end(),
					squared_weight_.begin(), _1*_1, queue);
				boost::compute::reduce(squared_weight_.begin(), squared_weight_.end(),
					&sum_of_squared, queue);
				if(sum_of_squared > c_*c_) {
					boost::compute::transform(weight_.begin(), weight_.end(),
						weight_.begin(), c_*_1/std::sqrt(sum_of_squared), queue);
				}
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(weight_, queue));
			}

		private:
			int input_dim_;
			int output_dim_;
			int batch_size_;
			scalar c_;
			gpu_vector weight_, transposed_weight_, squared_weight_;

			optimizer::any_optimizer optimizer_;
			kernel forward_kernel_, backward_kernel_, update_kernel_;
			gpu_vector input_, delta_;
			gpu_vector del_weight_;
		};
		template<typename WeightGen>
		decltype(auto) make_inner_product_wr(
				int input_dim,
				int output_dim,
				int batch_size,
				scalar c,
				WeightGen const& wg,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) {
			cpu_vector cpu_weight(input_dim*output_dim);
			std::generate(cpu_weight.begin(), cpu_weight.end(), wg);
			return neu::layer::inner_product_wr(input_dim, output_dim, batch_size, c,
				cpu_weight, optimizer, queue);
		}

		template<typename Rng>
		decltype(auto) make_inner_product_wr_xavier(
				int input_dim,
				int output_dim,
				int batch_size,
				scalar c,
				Rng&& rng,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue) {
			auto dist = std::normal_distribution<>(0.f, std::sqrt(1.f/input_dim));
			return neu::layer::make_inner_product_wr(input_dim, output_dim, batch_size, c,
				[&rng, dist]() mutable{ return dist(rng); },
				optimizer, queue);
		}

		namespace traits {
			template<>
			class serialize<inner_product_wr> {
			public:
				static decltype(auto) call(inner_product_wr const& ip,
						YAML::Emitter& emitter, boost::compute::command_queue& queue) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "inner_product_wr"
						<< YAML::Key << "input_dim"
							<< YAML::Value << neu::layer::input_dim(ip)
						<< YAML::Key << "output_dim"
							<< YAML::Value << neu::layer::output_dim(ip)
						<< YAML::Key << "batch_size"
							<< YAML::Value << neu::layer::batch_size(ip)
						<< YAML::Key << "c"
							<< YAML::Value << ip.c()
#ifndef NEU_LAYER_SERIALIZE_WITHOUT_LONG_VECTOR
						<< YAML::Key << "weight"
							<< YAML::Value << YAML::Flow << ip.weight(queue)
#endif
						<< YAML::Key << "optimizer"
							<< YAML::Value;
					optimizer::serialize(ip.optimizer(), emitter, queue);
					emitter << YAML::EndMap;
				}
			};
		}

		decltype(auto) deserialize_inner_product_wr(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			return inner_product_wr(
				node["input_dim"].as<int>(),
				node["output_dim"].as<int>(),
				node["batch_size"].as<int>(),
				node["c"].as<scalar>(),
				node["weight"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_INNER_PRODUCT_WR_LAYER_IMPL_HPP
