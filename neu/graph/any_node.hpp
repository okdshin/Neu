#ifndef NEU_GRAPH_ANY_NODE_HPP
#define NEU_GRAPH_ANY_NODE_HPP
//20150622
#include <cstddef>
#include <typeinfo>
#include <memory>
#include <functional>
#include <type_traits>
#include <neu/range/traits.hpp>
#include <neu/layer/traits.hpp>
#include <neu/range/gpu_buffer_range.hpp>
namespace neu {
	namespace graph {
		class any_node;
		namespace any_node_impl {
			class any_node_holder_base {
			public:
				any_node_holder_base() = default;
				any_node_holder_base(any_node_holder_base const&) = default;
				any_node_holder_base& operator=(any_node_holder_base const&) = default;
				any_node_holder_base(any_node_holder_base&&) = default;
				any_node_holder_base& operator=(any_node_holder_base&&) = default;
				virtual ~any_node_holder_base() noexcept = default;

				virtual std::type_info const& target_type() const = 0;
				virtual void* get() = 0;
				virtual void const* get() const = 0;

				virtual std::unique_ptr<any_node_holder_base> clone() = 0;

			};
			template<typename Node>
			class any_node_holder : public any_node_holder_base {
			private:
				template<typename T>
				struct unwrap_impl {
					using type = T;
					template<typename U>
					static decltype(auto) call(U& u) {
						return (u);
					}
				};
				template<typename T>
				struct unwrap_impl<std::reference_wrapper<T>> {
					using type = T;
					template<typename U>
					static decltype(auto) call(U& u) {
						return u.get();
					}
				};

				template<typename T>
				static decltype(auto) unwrap(T& t) {
					return unwrap_impl<std::decay_t<T>>::call(t);
				}
			public:
				any_node_holder() = delete;
				any_node_holder(any_node_holder const&) = default;
				any_node_holder& operator=(any_node_holder const&) = default;
				any_node_holder(any_node_holder&&) = default;
				any_node_holder& operator=(any_node_holder&&) = default;
				~any_node_holder() noexcept = default;

				explicit any_node_holder(Node const& l)
				: any_node_holder_base(), n_(l) {}
				
				std::type_info const& target_type() const override {
					return typeid(typename unwrap_impl<Node>::type);
				}

				void* get() override { return std::addressof(unwrap(n_)); }
				void const* get() const override { return std::addressof(unwrap(n_)); }

				std::unique_ptr<any_node_holder_base> clone() override {
					return std::make_unique<any_node_holder>(*this);
				}

			private:
				Node n_;
			};
		}

		class any_node {
		public:
			any_node() = default;
			any_node(any_node const& other) : holder_(other.holder_->clone()) {}
			any_node& operator=(any_node const& other) {
				auto h = other.holder_->clone();
				std::swap(holder_, h);
				return *this;
			}
			any_node(any_node&&) = default;
			any_node& operator=(any_node&&) = default;
			~any_node() noexcept = default;

			template<typename Node,
				typename = std::enable_if_t<
					!std::is_same<any_node, std::decay_t<Node>>::value>
			>
			any_node(Node&& l)
			: holder_(std::make_unique<any_node_impl::any_node_holder<
				std::decay_t<Node>>>(std::forward<Node>(l))) {}

			std::type_info const& target_type() const {
				return holder_->target_type();
			}

			bool operator==(std::nullptr_t) const noexcept {
				return holder_ == nullptr;
			}

			void swap(any_node& other) noexcept {
				std::swap(holder_, other.holder_);
			}

			template <typename Node>
			Node* target() {
				if(target_type() != typeid(Node)) {
					return nullptr;
				}
				else {
					return static_cast<Node*>(holder_->get());
				}
			}
			template <typename Node>
			Node const* target() const {
				return static_cast<Node const*>(
					const_cast<any_node*>(this)->target<Node>());
			}

		private:
			std::unique_ptr<any_node_impl::any_node_holder_base> holder_;
		};

		bool operator==(std::nullptr_t, any_node const& an) noexcept {
			return an == nullptr;
		}

		bool operator!=(any_node const& an, std::nullptr_t) noexcept {
			return !(an == nullptr);
		}
		bool operator!=(std::nullptr_t, any_node const& an) noexcept {
			return an != nullptr;
		}

		void swap(any_node& lhs, any_node& rhs) noexcept {
			lhs.swap(rhs);
		}
	}
}// namespace neu

#endif //NEU_GRAPH_ANY_NODE_HPP
