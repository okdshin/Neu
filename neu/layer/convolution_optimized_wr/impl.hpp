#ifndef NEU_LAYER_CONVOLUTION_WR_OPTIMIZED_IMPL_HPP
#define NEU_LAYER_CONVOLUTION_WR_OPTIMIZED_IMPL_HPP
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
#include <neu/layer/convolution_optimized_wr/kernel_source.hpp>
#include <neu/layer/impl/multiple_matrix_multiply.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>

namespace neu {
	namespace layer {
		class convolution_optimized_wr {
		public:
			convolution_optimized_wr() = default;

			convolution_optimized_wr(
				geometric_layer_property const& glp,
				int batch_size,
				scalar c,
				cpu_vector const& filters,
				optimizer::any_optimizer const& optimizer,
				boost::compute::command_queue& queue)
			: glp_(glp), batch_size_(batch_size), c_(c),
			filters_(filters.begin(), filters.end(), queue),
			squared_filters_(filters.size(),
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			optimizer_(optimizer),
			output_width_(output_width(glp)),
			forward_kernel_(make_kernel(convolution_optimized_wr_kernel_source,
				"forward", queue.get_context())),
			backward_kernel_(make_kernel(convolution_optimized_wr_kernel_source,
				"backward", queue.get_context())),
			update_kernel_(make_kernel(convolution_optimized_wr_kernel_source,
				"update_tile32",
				queue.get_context())),
			input_(layer::input_dim(glp)*batch_size_,
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
			delta_(layer::output_dim(glp)*batch_size_,
				std::numeric_limits<cl_float>::quiet_NaN(), queue),
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

				neu::layer::convolution_optimized_wr_forward(
					forward_kernel_,
					input, filters_, output,
					batch_size_,
					glp_.input_width, glp_.filter_width, output_width_,
					glp_.input_channel_num, glp_.output_channel_num,
					glp_.stride, glp_.pad, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input)
					== neu::layer::input_dim(glp_)*batch_size_);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				neu::range::copy(input, input_, queue);
				test_forward(batch_size_, input, output, queue);
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(delta)
					== static_cast<int>(delta_.size()));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				neu::range::copy(delta, delta_, queue);
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				backward_top(delta, queue);
				neu::layer::convolution_optimized_wr_backward(
					backward_kernel_,
					delta, filters_, prev_delta,
					batch_size_,
					glp_.input_width, glp_.filter_width, output_width_,
					glp_.input_channel_num, glp_.output_channel_num,
					glp_.stride, glp_.pad, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue& queue) {
				//neu::layer::convolution_optimized_wr_update_tile64_wpt2_reg(
				neu::layer::convolution_optimized_wr_update_tile32(
					update_kernel_,
					delta_, input_, del_filters_,
					batch_size_,
					glp_.input_width, glp_.filter_width, output_width_,
					glp_.input_channel_num, glp_.output_channel_num,
					glp_.stride, glp_.pad, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_filters_, queue));

				optimizer_.apply(filters_, del_filters_, queue);

				// weight regularization
				scalar sum_of_squared = 0.f;
				using boost::compute::lambda::_1;
				boost::compute::transform(filters_.begin(), filters_.end(),
					squared_filters_.begin(), _1*_1, queue);
				boost::compute::reduce(squared_filters_.begin(), squared_filters_.end(),
					&sum_of_squared, queue);
				if(sum_of_squared > c_*c_) {
					boost::compute::transform(filters_.begin(), filters_.end(),
						filters_.begin(), c_*_1/std::sqrt(sum_of_squared), queue);
				}
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(filters_, queue));
			}

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "convolution_optimized_wr";
				layer::serialize(glp_, emitter);
				emitter 
					<< YAML::Key << "output_width(calculated)"
						<< YAML::Value << output_width_
					<< YAML::Key << "batch_size"
						<< YAML::Value << batch_size_
					<< YAML::Key << "c"
						<< YAML::Value << c_
#ifndef NEU_LAYER_SERIALIZE_WITHOUT_LONG_VECTOR
					<< YAML::Key << "filters"
						<< YAML::Value << YAML::Flow << filters(queue)
#endif
					<< YAML::Key << "optimizer"
						<< YAML::Value;
				optimizer::serialize(optimizer_, emitter, queue);
				emitter << YAML::EndMap;
			}

		private:
			geometric_layer_property glp_;
			int batch_size_;
			scalar c_;

			gpu_vector filters_, squared_filters_;

			optimizer::any_optimizer optimizer_;

			int output_width_;

			kernel forward_kernel_;
			kernel backward_kernel_;
			kernel update_kernel_;

			gpu_vector input_;
			gpu_vector delta_;
			gpu_vector del_filters_;
		};

		template<typename FilterGen>
		decltype(auto) make_convolution_optimized_wr(
			geometric_layer_property const& glp,
			int batch_size, scalar c,
			FilterGen const& fg,
			optimizer::any_optimizer const& optimizer,
			boost::compute::command_queue& queue
		) {
			cpu_vector cpu_filters(neu::layer::filters_size(glp));
			std::generate(cpu_filters.begin(), cpu_filters.end(), fg);
			return convolution_optimized_wr(glp, batch_size, c, cpu_filters, optimizer, queue);
		}

		template<typename Rng>
		decltype(auto) make_convolution_optimized_wr_xavier(
			geometric_layer_property const& glp,
			int batch_size,
			scalar c,
			Rng&& rng,
			optimizer::any_optimizer const& optimizer,
			boost::compute::command_queue& queue
		) {
			auto dist = std::normal_distribution<neu::scalar>(
				0.f, std::sqrt(1.f/neu::layer::input_dim(glp)));
			return neu::layer::make_convolution_optimized_wr(glp, batch_size, c,
				[&rng, dist]() mutable { return dist(rng); },
				optimizer, queue);
		}

		decltype(auto) deserialize_convolution_optimized_wr(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			const auto glp = deserialize_geometric_layer_property(node);
			return convolution_optimized_wr(
				glp, node["batch_size"].as<int>(), node["c"].as<scalar>(),
				node["filters"].as<cpu_vector>(),
				optimizer::deserialize(node["optimizer"], queue),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_CONVOLUTION_OPTIMIZED_WR_IMPL_HPP
