#ifndef NEU_LAYER_ANY_LAYER_HPP
#define NEU_LAYER_ANY_LAYER_HPP
#include <type_traits>
#include <utility>
#include <memory>

#include <boost/compute/command_queue.hpp>
#include <neu/layer/rank_id.hpp>
#include <neu/range/gpu_buffer_range.hpp>
#include <yaml-cpp/yaml.h>
#include <neu/layer/traits.hpp>

namespace neu {
	namespace layer {
		namespace any_layer_impl {
			class any_layer_holder_base {
			public:
				any_layer_holder_base() = default;
				any_layer_holder_base(any_layer_holder_base const&) = default;
				any_layer_holder_base& operator=(any_layer_holder_base const&) = default;
				any_layer_holder_base& operator=(any_layer_holder_base&&) = default;
				virtual ~any_layer_holder_base() noexcept = default;

				virtual std::unique_ptr<any_layer_holder_base> clone() = 0;

				virtual int input_rank() const  = 0;
				virtual int output_rank() const  = 0;
				virtual int input_size(rank_id rid) const  = 0;
				virtual int output_size(rank_id rid) const  = 0;
				virtual int batch_size() const  = 0;
				virtual void test_forward(int batch_size, range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   = 0;
				virtual void forward(range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   = 0;
				virtual void backward_top(range::gpu_vector_range const& delta, boost::compute::command_queue& queue)   = 0;
				virtual void backward(range::gpu_vector_range const& delta, range::gpu_vector_range& prev_delta, boost::compute::command_queue& queue)   = 0;
				virtual void update(boost::compute::command_queue& queue)   = 0;
				virtual void serialize(YAML::Emitter& emitter, boost::compute::command_queue& queue) const  = 0;
			};

			template<typename T>
			class any_layer_holder : public any_layer_holder_base {
			public:
				any_layer_holder() = default;
				any_layer_holder(any_layer_holder const&) = default;
				any_layer_holder& operator=(any_layer_holder const&) = default;
				any_layer_holder& operator=(any_layer_holder&&) = default;
				~any_layer_holder() noexcept = default;

				std::unique_ptr<any_layer_holder_base> clone() override {
					return std::make_unique<any_layer_holder>(*this);
				}

				template<
					typename U,
					typename=std::enable_if_t<!std::is_same<any_layer_holder, std::decay_t<U>>::value>>
				explicit any_layer_holder(U&& u) 
					: any_layer_holder_base(), t_(std::forward<U>(u)) {}

				int input_rank() const  override {
					return ::neu::layer::input_rank(t_ );
				}
				int output_rank() const  override {
					return ::neu::layer::output_rank(t_ );
				}
				int input_size(rank_id rid) const  override {
					return ::neu::layer::input_size(t_, rid);
				}
				int output_size(rank_id rid) const  override {
					return ::neu::layer::output_size(t_, rid);
				}
				int batch_size() const  override {
					return ::neu::layer::batch_size(t_ );
				}
				void test_forward(int batch_size, range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   override {
					 ::neu::layer::test_forward(t_, batch_size, input, output, queue);
				}
				void forward(range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   override {
					 ::neu::layer::forward(t_, input, output, queue);
				}
				void backward_top(range::gpu_vector_range const& delta, boost::compute::command_queue& queue)   override {
					 ::neu::layer::backward_top(t_, delta, queue);
				}
				void backward(range::gpu_vector_range const& delta, range::gpu_vector_range& prev_delta, boost::compute::command_queue& queue)   override {
					 ::neu::layer::backward(t_, delta, prev_delta, queue);
				}
				void update(boost::compute::command_queue& queue)   override {
					 ::neu::layer::update(t_, queue);
				}
				void serialize(YAML::Emitter& emitter, boost::compute::command_queue& queue) const  override {
					 ::neu::layer::serialize(t_, emitter, queue);
				}

			private:
				T t_;
			};
		}

		class any_layer {
		public:
			any_layer() = default;
			any_layer(any_layer const& other) : holder_{other.holder_->clone()} {}
			any_layer& operator=(any_layer const& other) {
				auto h = other.holder_->clone();
				std::swap(holder_, h);
				return *this;
			}
			any_layer(any_layer&& other) = default;
			any_layer& operator=(any_layer&& other) = default;
			~any_layer() noexcept = default;

			template<
				typename U,
				typename=std::enable_if_t<!std::is_same<any_layer, std::decay_t<U>>::value>
			>
			any_layer(U&& u)
			: holder_(std::make_unique<any_layer_impl::any_layer_holder<std::decay_t<U>>>(
				std::forward<U>(u))) {}

			explicit operator bool() const noexcept {
				return static_cast<bool>(holder_);
			}

			decltype(auto) swap(any_layer& other) noexcept {
				holder_.swap(other.holder_);
			}

			int input_rank() const  {
				return holder_->input_rank();
			}
			int output_rank() const  {
				return holder_->output_rank();
			}
			int input_size(rank_id rid) const  {
				return holder_->input_size(rid);
			}
			int output_size(rank_id rid) const  {
				return holder_->output_size(rid);
			}
			int batch_size() const  {
				return holder_->batch_size();
			}
			void test_forward(int batch_size, range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   {
				 holder_->test_forward(batch_size, input, output, queue);
			}
			void forward(range::gpu_vector_range const& input, range::gpu_vector_range& output, boost::compute::command_queue& queue)   {
				 holder_->forward(input, output, queue);
			}
			void backward_top(range::gpu_vector_range const& delta, boost::compute::command_queue& queue)   {
				 holder_->backward_top(delta, queue);
			}
			void backward(range::gpu_vector_range const& delta, range::gpu_vector_range& prev_delta, boost::compute::command_queue& queue)   {
				 holder_->backward(delta, prev_delta, queue);
			}
			void update(boost::compute::command_queue& queue)   {
				 holder_->update(queue);
			}
			void serialize(YAML::Emitter& emitter, boost::compute::command_queue& queue) const  {
				 holder_->serialize(emitter, queue);
			}

		private:
			std::unique_ptr<any_layer_impl::any_layer_holder_base> holder_;
		};

		decltype(auto) swap(any_layer& lhs, any_layer& rhs) {
			lhs.swap(rhs);
		}
	}
}
#endif //NEU_LAYER_ANY_LAYER_HPP
