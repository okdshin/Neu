#ifndef NEU_NODE_ANY_NODE_HPP
#define NEU_NODE_ANY_NODE_HPP
#include <type_traits>
#include <utility>
#include <memory>

#include <boost/compute/command_queue.hpp>
#include <neu/range/gpu_buffer_range.hpp>
#include <neu/node/any_node_fwd.hpp>
#include <neu/node/traits.hpp>

namespace neu {
	namespace node {
		namespace any_node_impl {
			class any_node_holder_base {
			public:
				any_node_holder_base() = default;
				any_node_holder_base(any_node_holder_base const&) = default;
				any_node_holder_base& operator=(any_node_holder_base const&) = default;
				any_node_holder_base& operator=(any_node_holder_base&&) = default;
				virtual ~any_node_holder_base() noexcept = default;

				virtual std::unique_ptr<any_node_holder_base> clone() = 0;

				virtual void add_prev(any_node* node)   = 0;
				virtual void add_next(any_node* node)   = 0;
				virtual void forward(any_node* self, boost::compute::command_queue& queue)   = 0;
				virtual void backward(any_node* self, boost::compute::command_queue& queue)   = 0;
				virtual void update(boost::compute::command_queue& queue)   = 0;
				virtual range::gpu_vector_range output_for(any_node* next)   = 0;
				virtual range::gpu_vector_range next_delta_for(any_node* prev)   = 0;
			};

			template<typename T>
			class any_node_holder : public any_node_holder_base {
			public:
				any_node_holder() = default;
				any_node_holder(any_node_holder const&) = default;
				any_node_holder& operator=(any_node_holder const&) = default;
				any_node_holder& operator=(any_node_holder&&) = default;
				~any_node_holder() noexcept = default;

				std::unique_ptr<any_node_holder_base> clone() override {
					return std::make_unique<any_node_holder>(*this);
				}

				template<
					typename U,
					typename=std::enable_if_t<!std::is_same<any_node_holder, std::decay_t<U>>::value>>
				explicit any_node_holder(U&& u) 
					: any_node_holder_base(), t_(std::forward<U>(u)) {}

				void add_prev(any_node* node)   override {
					 ::neu::node::add_prev(t_, node);
				}
				void add_next(any_node* node)   override {
					 ::neu::node::add_next(t_, node);
				}
				void forward(any_node* self, boost::compute::command_queue& queue)   override {
					 ::neu::node::forward(t_, self, queue);
				}
				void backward(any_node* self, boost::compute::command_queue& queue)   override {
					 ::neu::node::backward(t_, self, queue);
				}
				void update(boost::compute::command_queue& queue)   override {
					 ::neu::node::update(t_, queue);
				}
				range::gpu_vector_range output_for(any_node* next)   override {
					return ::neu::node::output_for(t_, next);
				}
				range::gpu_vector_range next_delta_for(any_node* prev)   override {
					return ::neu::node::next_delta_for(t_, prev);
				}

			private:
				T t_;
			};
		}

		class any_node {
		public:
			any_node() = default;
			any_node(any_node const& other) : holder_{other.holder_->clone()} {}
			any_node& operator=(any_node const& other) {
				auto h = other.holder_->clone();
				std::swap(holder_, h);
				return *this;
			}
			any_node(any_node&& other) = default;
			any_node& operator=(any_node&& other) = default;
			~any_node() noexcept = default;

			template<
				typename U,
				typename=std::enable_if_t<!std::is_same<any_node, std::decay_t<U>>::value>
			>
			any_node(U&& u)
			: holder_(std::make_unique<any_node_impl::any_node_holder<std::decay_t<U>>>(
				std::forward<U>(u))) {}

			explicit operator bool() const noexcept {
				return static_cast<bool>(holder_);
			}

			decltype(auto) swap(any_node& other) noexcept {
				holder_.swap(other.holder_);
			}

			void add_prev(any_node* node)   {
				 holder_->add_prev(node);
			}
			void add_next(any_node* node)   {
				 holder_->add_next(node);
			}
			void forward(boost::compute::command_queue& queue)   {
				 holder_->forward(this, queue);
			}
			void backward(boost::compute::command_queue& queue)   {
				 holder_->backward(this, queue);
			}
			void update(boost::compute::command_queue& queue)   {
				 holder_->update(queue);
			}
			range::gpu_vector_range output_for(any_node* next)   {
				return holder_->output_for(next);
			}
			range::gpu_vector_range next_delta_for(any_node* prev)   {
				return holder_->next_delta_for(prev);
			}

		private:
			std::unique_ptr<any_node_impl::any_node_holder_base> holder_;
		};

		decltype(auto) swap(any_node& lhs, any_node& rhs) {
			lhs.swap(rhs);
		}
	}
}
#endif //NEU_NODE_ANY_NODE_HPP
