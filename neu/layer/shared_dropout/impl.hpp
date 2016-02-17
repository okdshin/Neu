#ifndef NEU_LAYER_SHARED_DROPOUT_IMPL_HPP
#define NEU_LAYER_SHARED_DROPOUT_IMPL_HPP
//20150217
#include <boost/compute/random/default_random_engine.hpp>
#include <boost/compute/random/uniform_real_distribution.hpp>
#include <boost/compute/functional/common.hpp>
#include <boost/compute/functional/bind.hpp>
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
#include <neu/kernel.hpp>
#include <neu/layer/geometric_layer_property.hpp>
#include <neu/layer/shared_dropout/kernel_source.hpp>
#include <neu/optimizer/deserialize.hpp>
#include <neu/optimizer/any_optimizer.hpp>

namespace neu {
	namespace layer {
		BOOST_COMPUTE_FUNCTION(float, shared_dropout_update_kernel,
				(float x, float probability), {
			return x < probability ? 1.f : 0.f;
		});
		class shared_dropout {
		public:
			shared_dropout() = default;

			shared_dropout(
				int batch_size,
				int input_dim,
				int shared_size,
				scalar probability,
				boost::compute::command_queue& queue)
			: batch_size_(batch_size),
			input_dim_(input_dim),
			shared_size_(shared_size),
			probability_(probability),
			mask_(input_dim_, queue.get_context()),
			engine_(queue), dist_(0.f, 1.f),
			dropout_test_forward_kernel_(make_kernel(dropout_test_forward_kernel_source,
				"test_forward", queue.get_context())),
			dropout_forward_kernel_(make_kernel(dropout_forward_kernel_source,
				"forward", queue.get_context())),
			dropout_backward_kernel_(make_kernel(dropout_backward_kernel_source,
				"backward", queue.get_context()))
			{
				if(input_dim%shared_size != 0) {
					throw "shared_size is wrong";
				}
				update(queue);
			}

			decltype(auto) batch_size() const { return batch_size_; }
			decltype(auto) shared_size() const { return shared_size_; }

			decltype(auto) input_rank() const { return 1; }
			decltype(auto) output_rank() const { return 1; }
			decltype(auto) input_size(rank_id ri) const {
				return ri == rank_id::dim ? input_dim_ : 0; }
			decltype(auto) output_size(rank_id ri) const {
				return input_size(ri); }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(int test_batch_size,
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(neu::range::distance(input) ==
					neu::layer::input_dim(*this)*test_batch_size);
				NEU_ASSERT(neu::range::distance(output) ==
					neu::layer::output_dim(*this)*test_batch_size);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				neu::enqueue_nd_range_kernel<2>(queue, dropout_test_forward_kernel_,
					{0, 0}, {input_dim_, test_batch_size},
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<int>(neu::range::get_begin_index(output)),
					static_cast<float>(probability_),
					static_cast<int>(input_dim_));

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));

				neu::enqueue_nd_range_kernel<2>(queue, dropout_forward_kernel_,
					{0, 0}, {input_dim_, batch_size_},
					neu::range::get_buffer(input),
					static_cast<int>(neu::range::get_begin_index(input)),
					neu::range::get_buffer(output),
					static_cast<int>(neu::range::get_begin_index(output)),
					mask_,
					static_cast<int>(input_dim_));

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			}

			template<typename InputRange>
			decltype(auto) backward_top(
					InputRange const& delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
				/* do nothing */
			}
			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange& prev_delta,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));

				neu::enqueue_nd_range_kernel<2>(queue, dropout_backward_kernel_,
					{0, 0}, {input_dim_, batch_size_},
					neu::range::get_buffer(prev_delta),
					static_cast<int>(neu::range::get_begin_index(prev_delta)),
					neu::range::get_buffer(delta),
					static_cast<int>(neu::range::get_begin_index(delta)),
					mask_,
					static_cast<int>(input_dim_));

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
			}

			void update(boost::compute::command_queue& queue) {
				dist_.generate(mask_.begin(), mask_.begin()+shared_size_, engine_, queue);
				boost::compute::transform(mask_.begin(), mask_.begin()+shared_size_,
					mask_.begin(),
					boost::compute::bind(shared_dropout_update_kernel,
						boost::compute::placeholders::_1, probability_), queue);
				//TODO dirty implementation, need to optimize
				for(int i = 1; i < static_cast<int>(mask_.size()/shared_size_); ++i) {
					boost::compute::copy(
						mask_.begin(), mask_.begin()+shared_size_,
						mask_.begin()+i*shared_size_);
				}

				NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(mask_, queue));
			}

			decltype(auto) serialize(
					YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				emitter << YAML::BeginMap
					<< YAML::Key << "layer_type"
						<< YAML::Value << "shared_dropout"
					<< YAML::Key << "input_dim"
						<< YAML::Value << input_dim_
					<< YAML::Key << "output_dim(equal to input_dim)"
						<< YAML::Value << input_dim_
					<< YAML::Key << "batch_size"
						<< YAML::Value << batch_size_
					<< YAML::Key << "probability"
						<< YAML::Value << probability_
					<< YAML::Key << "shared_size"
						<< YAML::Value << shared_size_
				<< YAML::EndMap;
			}

		private:
			int batch_size_;
			int input_dim_;
			int shared_size_;

			scalar probability_;

			gpu_vector mask_;
			boost::compute::default_random_engine engine_;
			boost::compute::uniform_real_distribution<float> dist_;

			kernel dropout_test_forward_kernel_;
			kernel dropout_forward_kernel_;
			kernel dropout_backward_kernel_;

		};

		decltype(auto) deserialize_shared_dropout(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			return shared_dropout(
				node["batch_size"].as<int>(),
				node["input_dim"].as<int>(),
				node["shared_size"].as<int>(),
				node["probability"].as<float>(),
				queue
			);
		}
	}
}// namespace neu

#endif //NEU_LAYER_SHARED_DROPOUT_IMPL_HPP
