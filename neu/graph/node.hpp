#ifndef NEU_GRAPH_NODE_HPP
#define NEU_GRAPH_NODE_HPP
//20151212
#include <map>
#include <neu/range/gpu_buffer_range.hpp>
#include <neu/layer/any_layer.hpp>
namespace neu {
	namespace graph {
		class node {
		public:
			virtual void add_prev(node*) = 0;
			virtual void add_next(node*) = 0;
			virtual void forward() = 0;
			virtual void backward() = 0;
			virtual void update() = 0;
			virtual range::gpu_vector_range output_for(node*) const = 0;
			virtual range::gpu_vector_range prev_delta_for(node*) const = 0;
		};
		decltype(auto) connect(node& from, node& to) {
			from.add_next(&to);
			to.add_prev(&from);
		}

		class fixed_data_node : public node {
		public:
			fixed_data_node() = default;

			fixed_data_node(cpu_vector const& data,
				boost::compute::command_queue& queue)
			: output_(data.begin(), data.end(), queue),
			output_range_(range::to_range(output_)) {}

			void add_prev(node* n) override {
				NEU_ASSERT(!"data node does not have any prev nodes");
			}
			void add_next(node* n) override {
				NEU_ASSERT(!next_);
				next_= n;
			}

			void forward() override {
				/* do nothing */
			}

			void backward() override {
				/* do nothing */
			}

			void update() override {
				/* do nothing */
			}

			range::gpu_vector_range output_for(node* next) const override {
				NEU_ASSERT(next == next_);
				return output_range_;
			}
			range::gpu_vector_range prev_delta_for(node* prev) const override {
				NEU_ASSERT(!"data node does not propagate delta");
				return range::gpu_vector_range();
			}

		private:
			gpu_vector output_;
			range::gpu_vector_range output_range_;

			node* prev_ = nullptr;
			node* next_ = nullptr;
		};

		class layer_node : public node {
		public:
			layer_node() = default;

			layer_node(layer::any_layer const& layer,
				boost::compute::command_queue& queue)
			: queue_(&queue), layer_(layer),
			output_(layer::whole_output_size(layer), queue.get_context()),
			output_range_(range::to_range(output_)),
			delta_(layer::whole_input_size(layer), queue.get_context()),
			delta_range_(range::to_range(delta_)) {}

			void add_prev(node* n) override {
				NEU_ASSERT(!prev_);
				prev_ = n;
			}
			void add_next(node* n) override {
				NEU_ASSERT(!next_);
				next_= n;
			}

			void forward() override {
				/*
				auto prev_out = prev_->output_for(this);
				gpu_vector prev_out_vec(
					range::begin(prev_out), range::end(prev_out), *queue_);
				std::cout << prev_out_vec[0] << std::endl;
				std::cout << prev_out_vec[1] << std::endl;
				std::cout << prev_out_vec[2] << std::endl;
				std::cout << prev_out_vec[3] << std::endl;
				std::cout << prev_out_vec[4] << std::endl;
				std::cout << prev_out_vec[5] << std::endl;
				std::cout << prev_out_vec[6] << std::endl;
				std::cout << prev_out_vec[7] << std::endl;
				*/
				//layer_.forward(prev_->output_for(this), output_range_, *queue_);
				cpu_vector cpu_output{0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
				gpu_vector output(cpu_output.begin(), cpu_output.end(), *queue_);
				auto r = range::to_range(output);
				layer_.forward(r, output_range_, *queue_);
			}

			void backward() override {
				layer_.backward(next_->prev_delta_for(this), delta_range_, *queue_);
			}

			void update() override {
				layer_.update(*queue_);
			}

			range::gpu_vector_range output_for(node* next) const override {
				NEU_ASSERT(next == next_);
				return output_range_;
			}
			range::gpu_vector_range prev_delta_for(node* prev) const override {
				NEU_ASSERT(prev == prev_);
				return delta_range_;
			}

		private:
			boost::compute::command_queue* queue_ = nullptr;
			layer::any_layer layer_{};

			gpu_vector output_;
			range::gpu_vector_range output_range_;

			gpu_vector delta_;
			range::gpu_vector_range delta_range_;

			node* prev_ = nullptr;
			node* next_ = nullptr;
		};
		/*
		class layer_node {
		public:
			decltype(auto) connect(node& next) {
				next_list_.push_back(&next);
				next.add_prev(this);
			}
			decltype(auto) add_prev(node* prev) {
				NEU_ASSERT(prev);
				prev_list.push_back(prev);
			}

			bool is_forwarded() const {
				return is_forwarded_;
			}

			bool is_forwardable() const {
				return prev_->is_forwarded();
			}

			decltype(auto) forward() {
				NEU_ASSERT(is_forwardable());
				layer_.forward(prev_->output(this), output_, queue_);
				is_forwarded_ = true;
			}

			decltype(auto) backward() {

			}

			decltype(auto) update() {
			}

			decltype(auto) reset() {
				is_forwarded_ = false;
			}

		private:
			node* prev_;
			std::vector<node*> next_list_;
			layer::any_layer layer_;

			bool is_forwarded_ = false;
		};

		class concat_node {
		public:
			decltype(auto) connect(node* next) {
				NEU_ASSERT(next);
				next_list_.push_back(next);
				next->add_prev(this);
			}
			decltype(auto) add_prev(node* prev) {
				NEU_ASSERT(prev);
				prev_list.push_back(prev);
			}

			bool is_forwarded() const { return is_forwarded_; }

			bool is_forwardable() const {
				return std::all_of(prev_list_.begin(), prev_list_.end(),
					[](auto& n){ return n->is_forwarded(); });
			}

			decltype(auto) forward() {
				NEU_ASSERT(is_forwardable());
				for(auto prev : prev_list_) {
					auto const& output = prev->output();
					boost::compute::copy(output.begin(), output.end(),
						output_.end(), queue_);
				}
				is_forwarded_ = true;
			}
			decltype(auto) output(node* receiver) const {
				return (output_);
			}

			bool is_backwarded() const { return is_backwarded_; }

			bool is_backwardable() const {
				return next_->is_backwarded();
			}

			decltype(auto) backward() {
			}
			decltype(auto) delta(node* receiver) const {
				auto iter = delta_map_.find(receiver);
				NEU_ASSERT(iter == delta_map_.end());
				return *iter;
			}

			decltype(auto) update() {
			}

			decltype(auto) reset() {
				is_forwarded_ = false;
			}

		private:
			std::vector<node*> prev_list_;
			node* next_;
			layer::any_layer layer_;

			bool is_forwarded_ = false;

			std::map<node*, gpu_vector> delta_map_;
		};
		*/
	}
}// namespace neu

#endif //NEU_GRAPH_NODE_HPP
