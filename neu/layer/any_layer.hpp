#ifndef NEU_LAYER_ANY_LAYER_HPP
#define NEU_LAYER_ANY_LAYER_HPP
//20150622
#include <typeinfo>
#include <memory>
#include <functional>
#include <type_traits>
#include <neu/range/traits.hpp>
#include <neu/layer/traits.hpp>
#include <neu/range/gpu_buffer_range.hpp>
namespace neu {
	namespace layer {
		class any_layer;
		namespace any_layer_impl {
			class any_layer_holder_base {
			public: any_layer_holder_base() = default;
				any_layer_holder_base(any_layer_holder_base const&) = default;
				any_layer_holder_base& operator=(any_layer_holder_base const&) = default;
				any_layer_holder_base(any_layer_holder_base&&) = default;
				any_layer_holder_base& operator=(any_layer_holder_base&&) = default;
				virtual ~any_layer_holder_base() noexcept = default;

				virtual std::type_info const& target_type() const = 0;
				virtual void* get() = 0;
				virtual void const* get() const = 0;

				virtual std::unique_ptr<any_layer_holder_base> clone() = 0;

				virtual std::size_t input_dim() const = 0;
				virtual std::size_t output_dim() const = 0;
				virtual std::size_t batch_size() const = 0;

				virtual void test_forward(std::size_t batch_size,
					range::gpu_vector_range const& input,
					range::gpu_vector_range const& output,
					boost::compute::command_queue& queue) = 0;
				virtual void forward(range::gpu_vector_range const& input,
					range::gpu_vector_range const& output,
					boost::compute::command_queue& queue) = 0;

				virtual void backward(range::gpu_vector_range const& delta,
					range::gpu_vector_range const& prev_delta,
					bool is_top,
					boost::compute::command_queue& queue) = 0;

				virtual void update(boost::compute::command_queue& queue) = 0;

				virtual void save(YAML::Emitter& emitter,
					boost::compute::command_queue& queue) const = 0;

			};
			template<typename Layer>
			class any_layer_holder : public any_layer_holder_base {
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
				any_layer_holder() = delete;
				any_layer_holder(any_layer_holder const&) = default;
				any_layer_holder& operator=(any_layer_holder const&) = default;
				any_layer_holder(any_layer_holder&&) = default;
				any_layer_holder& operator=(any_layer_holder&&) = default;
				~any_layer_holder() noexcept = default;

				explicit any_layer_holder(Layer const& l)
				: any_layer_holder_base(), l_(l) {}
				
				std::type_info const& target_type() const override {
					return typeid(typename unwrap_impl<Layer>::type);
				}

				void* get() override { return std::addressof(unwrap(l_)); }
				void const* get() const override { return std::addressof(unwrap(l_)); }

				std::unique_ptr<any_layer_holder_base> clone() override {
					return std::make_unique<any_layer_holder>(*this);
				}

				std::size_t input_dim() const override {
					return neu::layer::input_dim(unwrap(l_));
				}
				std::size_t output_dim() const override {
					return neu::layer::output_dim(unwrap(l_));
				}
				std::size_t batch_size() const override {
					return neu::layer::batch_size(unwrap(l_));
				}

				void test_forward(std::size_t batch_size,
						range::gpu_vector_range const& input,
						range::gpu_vector_range const& output,
						boost::compute::command_queue& queue) override {
					neu::layer::test_forward(unwrap(l_), batch_size, input, output, queue);
				}
				void forward(range::gpu_vector_range const& input,
						range::gpu_vector_range const& output,
						boost::compute::command_queue& queue) override {
					neu::layer::forward(unwrap(l_), input, output, queue);
				}

				void backward(range::gpu_vector_range const& delta,
						range::gpu_vector_range const& prev_delta,
						bool is_top,
						boost::compute::command_queue& queue) override {
					neu::layer::backward(unwrap(l_), delta, prev_delta, is_top, queue);
				}

				void update(boost::compute::command_queue& queue) override {
					neu::layer::update(unwrap(l_), queue);
				}

				void save(YAML::Emitter& emitter,
						boost::compute::command_queue& queue) const override {
					neu::layer::save(unwrap(l_), emitter, queue);
				}
				
			private:
				Layer l_;
			};
		}

		class any_layer {
		public:
			any_layer() = default;
			any_layer(any_layer const& other) : holder_(other.holder_->clone()) {}
			any_layer& operator=(any_layer const& other) {
				auto h = other.holder_->clone();
				std::swap(holder_, h);
				return *this;
			}
			any_layer(any_layer&&) = default;
			any_layer& operator=(any_layer&&) = default;
			~any_layer() noexcept = default;

			template<typename Layer,
				typename = std::enable_if_t<
					!std::is_same<any_layer, std::decay_t<Layer>>::value>
			>
			any_layer(Layer&& l)
			: holder_(std::make_unique<any_layer_impl::any_layer_holder<
				std::decay_t<Layer>>>(std::forward<Layer>(l))) {}

			std::type_info const& target_type() const {
				return holder_->target_type();
			}

			bool operator==(nullptr_t) const noexcept {
				return holder_ == nullptr;
			}

			void swap(any_layer& other) noexcept {
				std::swap(holder_, other.holder_);
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
				return static_cast<Layer const*>(const_cast<any_layer*>(this)->target<Layer>());
			}

			std::size_t input_dim() const { return holder_->input_dim(); }
			std::size_t output_dim() const { return holder_->output_dim(); }
			std::size_t batch_size() const { return holder_->batch_size(); }

			void test_forward(std::size_t batch_size,
					range::gpu_vector_range const& input,
					range::gpu_vector_range const& output,
					boost::compute::command_queue& queue) {
				holder_->test_forward(batch_size, input, output, queue);
			}
			void forward(range::gpu_vector_range const& input,
					range::gpu_vector_range const& output,
					boost::compute::command_queue& queue) {
				holder_->forward(input, output, queue);
			}

			void backward(range::gpu_vector_range const& delta,
					range::gpu_vector_range const& prev_delta,
					bool is_top,
					boost::compute::command_queue& queue) {
				holder_->backward(delta, prev_delta, is_top, queue);
			}

			void update(boost::compute::command_queue& queue) {
				holder_->update(queue);
			}

			void save(YAML::Emitter& emitter, boost::compute::command_queue& queue) const {
				holder_->save(emitter, queue);
			}

		private:
			std::unique_ptr<any_layer_impl::any_layer_holder_base> holder_;
		};

		bool operator==(std::nullptr_t, any_layer const& al) noexcept {
			return al == nullptr;
		}

		bool operator!=(any_layer const& al, std::nullptr_t) noexcept {
			return !(al == nullptr);
		}
		bool operator!=(std::nullptr_t, any_layer const& al) noexcept {
			return al != nullptr;
		}

		void swap(any_layer& lhs, any_layer& rhs) noexcept {
			lhs.swap(rhs);
		}
	}
}// namespace neu

#endif //NEU_LAYER_ANY_LAYER_HPP
