#ifndef NEU_NODE_FIXED_DATA_HPP
#define NEU_NODE_FIXED_DATA_HPP
//20160108
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	namespace node {
		class fixed_data {
		public:
			fixed_data() = default;

			fixed_data(cpu_vector const& data,
				boost::compute::command_queue& queue)
			: output_(data.begin(), data.end(), queue) {}

			void add_prev(any_node* n) {
				NEU_ASSERT(!"data node does not have any prev nodes");
			}
			void add_next(any_node* n) {
				NEU_ASSERT(!next_);
				next_= n;
			}

			void forward(any_node*, boost::compute::command_queue&) {
				/* do nothing */
			}

			void backward(any_node*, boost::compute::command_queue&) {
				/* do nothing */
			}

			void update(boost::compute::command_queue&) {
				/* do nothing */
			}

			range::gpu_vector_range output_for(any_node* next) const {
				NEU_ASSERT(next == next_);
				return range::to_range(output_);
			}
			range::gpu_vector_range next_delta_for(any_node* prev) const {
				NEU_ASSERT(!"data node does not propagate delta");
				return range::gpu_vector_range();
			}

		private:
			gpu_vector output_;

			any_node* prev_ = nullptr;
			any_node* next_ = nullptr;
		};

		class error_for_fixed_data {
		public:
			error_for_fixed_data() = default;

			error_for_fixed_data(cpu_vector const& data,
				boost::compute::command_queue& queue)
			: teach_(data.begin(), data.end(), queue),
			error_(data.size(), queue.get_context()) {}

			void add_prev(any_node* n) {
				NEU_ASSERT(!prev_);
				prev_ = n;
			}
			void add_next(any_node* n) {
				NEU_ASSERT(!"error node does not have any next nodes");
			}

			void forward(any_node*, boost::compute::command_queue&) {
				/* do nothing */
			}

			void backward(any_node* self, boost::compute::command_queue& queue) {	
				auto teach_range = range::to_range(teach_);
				range::calc_last_layer_delta(
					prev_->output_for(self), teach_range, error_, queue);
			}

			void update(boost::compute::command_queue&) {
				/* do nothing */
			}

			range::gpu_vector_range output_for(any_node* next) const {
				NEU_ASSERT(!"data node does not propagate input");
				return range::gpu_vector_range();
			}
			range::gpu_vector_range next_delta_for(any_node* prev) const {
				NEU_ASSERT(prev == prev_);
				return range::to_range(error_);
			}

		private:
			gpu_vector teach_;
			gpu_vector error_;

			any_node* prev_ = nullptr;
		};
	}
}// namespace neu

#endif //NEU_GRAPH_FIXED_DATA_NODE_HPP
