#ifndef NEU_LAYER_HPP
#define NEU_LAYER_HPP
//20150622
#include <typeinfo>
#include <memory>
#include <functional>
#include <type_traits>
#include <neu/basic_type.hpp>
#include <neu/layer_traits.hpp>
namespace neu {
	class layer;
	namespace layer_impl {
		class layer_holder_base {
		public:
			layer_holder_base() = default;
			layer_holder_base(layer_holder_base const&) = default;
			layer_holder_base& operator=(layer_holder_base const&) = default;
			layer_holder_base(layer_holder_base&&) = default;
			layer_holder_base& operator=(layer_holder_base&&) = default;
			virtual ~layer_holder_base() {}

			virtual std::type_info const& target_type() const = 0;
			virtual void* get() = 0;

			virtual std::unique_ptr<layer_holder_base> clone() = 0;

			virtual void test_forward(gpu_vector const& input) = 0;
			virtual void forward(gpu_vector const& input) = 0;
			virtual gpu_vector const& get_next_input() const = 0;

			virtual void backward(gpu_vector const& delta) = 0;
			virtual gpu_vector const& get_prev_delta() const = 0;

			virtual bool should_update() const = 0;
			virtual void update() = 0;

		};
		template<typename Layer>
		class layer_holder : public layer_holder_base {
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
			layer_holder() = delete;
			layer_holder(layer_holder const&) = default;
			layer_holder& operator=(layer_holder const&) = default;
			layer_holder(layer_holder&&) = default;
			layer_holder& operator=(layer_holder&&) = default;
			~layer_holder() = default;

			explicit layer_holder(Layer const& l)
			: layer_holder_base(), l_(l) {}
			
			std::type_info const& target_type() const override {
				return typeid(typename unwrap_impl<Layer>::type);
			}

			void* get() override { return std::addressof(unwrap(l_)); }

			std::unique_ptr<layer_holder_base> clone() override {
				return std::make_unique<layer_holder>(*this);
			}

			void test_forward(gpu_vector const& input) override {
				layer_test_forward(unwrap(l_), input);
			}
			void forward(gpu_vector const& input) override {
				layer_forward(unwrap(l_), input);
			}
			gpu_vector const& get_next_input() const override {
				return layer_get_next_input(unwrap(l_));
			}

			void backward(gpu_vector const& delta) override {
				layer_backward(unwrap(l_), delta);
			}
			gpu_vector const& get_prev_delta() const override {
				return layer_get_prev_delta(unwrap(l_));
			}

			bool should_update() const override {
				return layer_should_update(unwrap(l_));
			}
			void update() override {
				layer_update(unwrap(l_));
			}
			
		private:
			Layer l_;
		};
	}

	class layer {
	public:
		layer() = delete;
		layer(layer const& other) : holder_(other.holder_->clone()) {}
		layer& operator=(layer const& other) {
			holder_ = other.holder_->clone();
			return *this;
		}
		layer(layer&&) = default;
		layer& operator=(layer&&) = default;
		~layer() = default;

		template<typename Layer,
			typename = std::enable_if_t<!std::is_same<layer, std::decay_t<Layer>>::value>
		>
		layer(Layer&& l)
		: holder_(std::make_unique<layer_impl::layer_holder<std::decay_t<Layer>>>(
					std::forward<Layer>(l))) {}

		std::type_info const& target_type() const {
			return holder_->target_type();
		}

		template <typename Layer>
		Layer* target() {
			if(target_type() != typeid(Layer)) {
				return nullptr;
			}
			else {
				return static_cast<Layer*>(holder_->get());
			}
		}
		template <typename Layer>
		Layer const* target() const {
			return static_cast<Layer const*>(const_cast<layer*>(this)->target<Layer>());
		}

		void test_forward(gpu_vector const& input) {
			holder_->test_forward(input);
		}
		void forward(gpu_vector const& input) {
			holder_->forward(input);
		}
		gpu_vector const& get_next_input() const {
			return holder_->get_next_input();
		}

		void backward(gpu_vector const& delta) {
			holder_->backward(delta);
		}
		gpu_vector const& get_prev_delta() const {
			return holder_->get_prev_delta();
		}

		bool should_update() const {
			return holder_->should_update();
		}
		void update() {
			holder_->update();
		}

	private:
		std::unique_ptr<layer_impl::layer_holder_base> holder_;
	};
}// namespace neu

namespace neu_layer_traits {
	template<>
	class should_update<neu::layer> {
	public:
		static bool call(neu::layer const& l) {
			return l.should_update();
		}
	};
}

#endif //NEU_LAYER_HPP
