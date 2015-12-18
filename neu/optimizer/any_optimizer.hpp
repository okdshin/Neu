#ifndef NEU_OPTIMIZER_ANY_OPTIMIZER_HPP
#define NEU_OPTIMIZER_ANY_OPTIMIZER_HPP
//20150622
#include <typeinfo>
#include <memory>
#include <functional>
#include <type_traits>
#include <neu/range/traits.hpp>
#include <neu/optimizer/traits.hpp>
#include <neu/range/gpu_buffer_range.hpp>
namespace neu {
	namespace optimizer {
		class any_optimizer;
		namespace any_optimizer_impl {
			class any_optimizer_holder_base {
			public:
				any_optimizer_holder_base() = default;
				any_optimizer_holder_base(any_optimizer_holder_base const&) = default;
				any_optimizer_holder_base& operator=(
					any_optimizer_holder_base const&) = default;
				any_optimizer_holder_base(any_optimizer_holder_base&&) = default;
				any_optimizer_holder_base& operator=(
					any_optimizer_holder_base&&) = default;
				virtual ~any_optimizer_holder_base() noexcept = default;

				virtual std::type_info const& target_type() const = 0;
				virtual void* get() = 0;

				virtual std::unique_ptr<any_optimizer_holder_base> clone() = 0;

				virtual void apply(gpu_vector& weight, gpu_vector const& del_weight,
					boost::compute::command_queue& queue) = 0;

				virtual void serialize(YAML::Emitter& emitter,
					boost::compute::command_queue& queue) const = 0;

			};
			template<typename Optimizer>
			class any_optimizer_holder : public any_optimizer_holder_base {
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
				any_optimizer_holder() = default;
				any_optimizer_holder(any_optimizer_holder const&) = default;
				any_optimizer_holder& operator=(any_optimizer_holder const&) = default;
				any_optimizer_holder(any_optimizer_holder&&) = default;
				any_optimizer_holder& operator=(any_optimizer_holder&&) = default;
				~any_optimizer_holder() noexcept = default;

				explicit any_optimizer_holder(Optimizer const& l)
				: any_optimizer_holder_base(), l_(l) {}
				
				std::type_info const& target_type() const override {
					return typeid(typename unwrap_impl<Optimizer>::type);
				}

				void* get() override { return std::addressof(unwrap(l_)); }

				std::unique_ptr<any_optimizer_holder_base> clone() override {
					return std::make_unique<any_optimizer_holder>(*this);
				}

				void apply(gpu_vector& weight, gpu_vector const& del_weight,
						boost::compute::command_queue& queue) override {
					neu::optimizer::apply(unwrap(l_), weight, del_weight, queue);
				}

				void serialize(YAML::Emitter& emitter,
						boost::compute::command_queue& queue) const override {
					neu::optimizer::serialize(unwrap(l_), emitter, queue);
				}
				
			private:
				Optimizer l_;
			};
		}

		class any_optimizer {
		public:
			any_optimizer() = default;
			any_optimizer(any_optimizer const& other) : holder_(other.holder_->clone()) {}
			any_optimizer& operator=(any_optimizer const& other) {
				auto h = other.holder_->clone();
				std::swap(holder_, h);
				return *this;
			}
			any_optimizer(any_optimizer&&) = default;
			any_optimizer& operator=(any_optimizer&&) = default;
			~any_optimizer() noexcept = default;

			template<typename Optimizer,
				typename = std::enable_if_t<!std::is_same<
					any_optimizer, std::decay_t<Optimizer>>::value>
			>
			any_optimizer(Optimizer&& l)
			: holder_(std::make_unique<any_optimizer_impl::any_optimizer_holder<
				std::decay_t<Optimizer>>>(std::forward<Optimizer>(l))) {}

			std::type_info const& target_type() const {
				return holder_->target_type();
			}

			template <typename Optimizer>
			Optimizer* target() {
				if(target_type() != typeid(Optimizer)) {
					return nullptr;
				}
				else {
					return static_cast<Optimizer*>(holder_->get());
				}
			}
			template <typename Optimizer>
			Optimizer const* target() const {
				return static_cast<Optimizer const*>(
					const_cast<any_optimizer*>(this)->target<Optimizer>());
			}

			void apply(gpu_vector& weight, gpu_vector const& del_weight,
					boost::compute::command_queue& queue) {
				holder_->apply(weight, del_weight, queue);
			}

			void serialize(YAML::Emitter& emitter,
					boost::compute::command_queue& queue) const {
				holder_->serialize(emitter, queue);
			}

		private:
			std::unique_ptr<any_optimizer_impl::any_optimizer_holder_base> holder_;
		};
	}
}// namespace neu

#endif //NEU_OPTIMIZER_ANY_OPTIMIZER_HPP
