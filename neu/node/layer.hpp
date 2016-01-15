#ifndef NEU_NODE_LAYER_HPP
#define NEU_NODE_LAYER_HPP
//20160108
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/gpu_buffer_range.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/node/any_node.hpp>
#include <neu/layer/any_layer.hpp>
namespace neu {
	namespace node {
		class layer {
		public:
			layer() = default;

			layer(neu::layer::any_layer const& layer,
				boost::compute::command_queue& queue)
			: layer_(layer),
			output_(neu::layer::whole_output_size(layer), queue.get_context()),
			delta_(neu::layer::whole_input_size(layer), queue.get_context()) {}

			void add_prev(any_node* n) {
				NEU_ASSERT(!prev_);
				prev_ = n;
			}
			void add_next(any_node* n) {
				NEU_ASSERT(!next_);
				next_= n;
			}

			void forward(any_node* self, boost::compute::command_queue& queue) {
				auto output_range = range::to_range(output_);
				layer_.forward(prev_->output_for(self), output_range, queue);
			}

			void backward(any_node* self, boost::compute::command_queue& queue) {
				auto delta_range = range::to_range(delta_);
				layer_.backward(next_->next_delta_for(self), delta_range, queue);
			}

			void update(boost::compute::command_queue& queue) {
				layer_.update(queue);
			}

			range::gpu_vector_range output_for(any_node* next) const {
				NEU_ASSERT(next == next_);
				return range::to_range(output_);
			}
			range::gpu_vector_range next_delta_for(any_node* prev) const {
				NEU_ASSERT(prev == prev_);
				return range::to_range(delta_);
			}

		private:
			neu::layer::any_layer layer_{};

			gpu_vector output_;
			gpu_vector delta_;

			any_node* prev_ = nullptr;
			any_node* next_ = nullptr;
		};
	}
}// namespace neu

#endif //NEU_NODE_LAYER_HPP
