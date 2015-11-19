#ifndef NEU_LAYER_HPP
#define NEU_LAYER_HPP
//20150622
#include <typeinfo>
#include <memory>
#include <functional>
#include <type_traits>
#include <neu/range_traits.hpp>
#include <neu/layer_traits.hpp>
#include <neu/gpu_buffer_range.hpp>
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
			virtual ~layer_holder_base() = default;

			virtual std::type_info const& target_type() const = 0;
			virtual void* get() = 0;

			virtual std::unique_ptr<layer_holder_base> clone() = 0;

			virtual std::size_t input_dim() const = 0;
			virtual std::size_t output_dim() const = 0;
			virtual std::size_t batch_size() const = 0;

			virtual void test_forward(std::size_t batch_size,
				gpu_vector_range const& input, gpu_vector_range const& output,
				boost::compute::command_queue& queue) = 0;
			virtual void forward(gpu_vector_range const& input,
				gpu_vector_range const& output,
				boost::compute::command_queue& queue) = 0;

			virtual void backward(gpu_vector_range const& delta,
				gpu_vector_range const& prev_delta,
				boost::compute::command_queue& queue) = 0;

			virtual bool should_update() const = 0;
			virtual void update(boost::compute::command_queue& queue) = 0;

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

			std::size_t input_dim() const override {
				return neu::layer_input_dim(unwrap(l_));
			}
			std::size_t output_dim() const override {
				return neu::layer_output_dim(unwrap(l_));
			}
			std::size_t batch_size() const override {
				return neu::layer_batch_size(unwrap(l_));
			}

			void test_forward(std::size_t batch_size, gpu_vector_range const& input,
					gpu_vector_range const& output,
					boost::compute::command_queue& queue) override {
				neu::layer_test_forward(unwrap(l_), batch_size, input, output, queue);
			}
			void forward(gpu_vector_range const& input,
					gpu_vector_range const& output,
					boost::compute::command_queue& queue) override {
				neu::layer_forward(unwrap(l_), input, output, queue);
			}

			void backward(gpu_vector_range const& delta,
					gpu_vector_range const& prev_delta,
					boost::compute::command_queue& queue) override {
				neu::layer_backward(unwrap(l_), delta, prev_delta, queue);
			}

			bool should_update() const override {
				return layer_should_update(unwrap(l_));
			}
			void update(boost::compute::command_queue& queue) override {
				neu::layer_update(unwrap(l_), queue);
			}
			
		private:
			Layer l_;
		};
	}

	class layer {
	public:
		layer() = default;
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

		std::size_t input_dim() const { return holder_->input_dim(); }
		std::size_t output_dim() const { return holder_->output_dim(); }
		std::size_t batch_size() const { return holder_->batch_size(); }

		void test_forward(std::size_t batch_size, gpu_vector_range const& input,
				gpu_vector_range const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			holder_->test_forward(batch_size, input, output, queue);
		}
		void forward(gpu_vector_range const& input,
				gpu_vector_range const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			holder_->forward(input, output, queue);
		}

		void backward(gpu_vector_range const& delta,
				gpu_vector_range const& prev_delta,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			holder_->backward(delta, prev_delta, queue);
		}

		bool should_update() const {
			return holder_->should_update();
		}
		void update(
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			holder_->update(queue);
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
