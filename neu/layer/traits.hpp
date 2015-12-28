#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20151025
#include <type_traits>
#include <exception>
#include <string>
#include <boost/compute/command_queue.hpp>
#include <yaml-cpp/yaml.h>

namespace neu {
	namespace layer {
		enum class rank_id : std::size_t {
			dim = 0,
			width = 0,
			height = 1,
			channel_num = 2
		};

		//
		// input_rank
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class input_rank {
			public:
				static decltype(auto) call(Layer const& l) {
					return l.input_rank();
				}
			};
		}
		template<typename Layer>
		decltype(auto) input_rank(Layer const& l) {
			return ::neu::layer::traits::input_rank<std::decay_t<Layer>>::call(l);
		}

		//
		// output_rank
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class output_rank {
			public:
				static decltype(auto) call(Layer const& l) {
					return l.output_rank();
				}
			};
		}
		template<typename Layer>
		decltype(auto) output_rank(Layer const& l) {
			return ::neu::layer::traits::output_rank<std::decay_t<Layer>>::call(l);
		}

		class invalid_rank_access_error : public std::exception {
		public:
			invalid_rank_access_error(std::string const& message)
				: message_(message) {}

			const char* what() const noexcept override {
				return ("invalid_rank_access_error: "+message_).c_str();
			}

		private:
			std::string message_;
		};

		//
		// input_size
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class input_size {
			public:
				static decltype(auto) call(Layer const& l, rank_id ri) {
					return l.input_size(ri);
				}
			};
		}
		template<typename Layer>
		decltype(auto) input_size(Layer const& l, rank_id ri) {
			if(static_cast<int>(ri) >= input_rank(l)) {
				throw invalid_rank_access_error("out of range of input rank");
			}
			return ::neu::layer::traits::input_size<std::decay_t<Layer>>::call(l, ri);
		}

		// helper for geometric layers
		template<typename Layer>
		decltype(auto) input_width(Layer const& l) {
			if(input_rank(l) <= 1) {
				throw invalid_rank_access_error("input rank should be greater than 1");
			}
			return ::neu::layer::input_size(l, rank_id::width);
		}

		//
		// output_size
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class output_size {
			public:
				static decltype(auto) call(Layer const& l, rank_id ri) {
					return l.output_size(ri);
				}
			};
		}
		template<typename Layer>
		decltype(auto) output_size(Layer const& l, rank_id ri) {
			if(static_cast<int>(ri) >= output_rank(l)) {
				throw invalid_rank_access_error("out of range of output rank");
			}
			return ::neu::layer::traits::output_size<std::decay_t<Layer>>::call(l, ri);
		}

		// helper for geometric layers
		template<typename Layer>
		decltype(auto) output_width(Layer const& l) {
			if(output_rank(l) <= 1) {
				throw invalid_rank_access_error("output rank should be greater than 1");
			}
			return ::neu::layer::output_size(l, rank_id::width);
		}

		template<typename Layer>
		decltype(auto) output_channel_num(Layer const& l) {
			if(output_rank(l) <= 2) {
				throw invalid_rank_access_error("output rank should be greater than 1");
			}
			return ::neu::layer::output_size(l, rank_id::channel_num);
		}

		// input_dim
		template<typename Layer>
		decltype(auto) input_dim(Layer const& l) {
			auto dim = 1;
			for(auto i = 0; i < static_cast<int>(::neu::layer::input_rank(l)); ++i) {
				dim *= input_size(l, static_cast<rank_id>(i));
			}
			return dim;
		}

		// output_dim
		template<typename Layer>
		decltype(auto) output_dim(Layer const& l) {
			auto dim = 1;
			for(auto i = 0; i < static_cast<int>(::neu::layer::output_rank(l)); ++i) {
				dim *= output_size(l, static_cast<rank_id>(i));
			}
			return dim;
		}


		//
		// batch_size
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class batch_size {
			public:
				static decltype(auto) call(Layer const& l) {
					return l.batch_size();
				}
			};
		}
		template<typename Layer>
		decltype(auto) batch_size(Layer const& l) {
			return ::neu::layer::traits::batch_size<std::decay_t<Layer>>::call(l);
		}

		// input_size
		template<typename Layer>
		decltype(auto) whole_input_size(Layer const& l) {
			return ::neu::layer::input_dim(l)*::neu::layer::batch_size(l);
		}

		// output_size
		template<typename Layer>
		decltype(auto) whole_output_size(Layer const& l) {
			return ::neu::layer::output_dim(l)*::neu::layer::batch_size(l);
		}

		//
		// test_forward
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class test_forward {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(Layer& l, std::size_t batch_size,
						InputRange const& input, OutputRange& output,
						boost::compute::command_queue& queue) {
					l.test_forward(batch_size, input, output, queue);
				}
			};
		}
		template<typename Layer, typename InputRange, typename OutputRange>
		decltype(auto) test_forward(Layer& l, std::size_t batch_size,
				InputRange const& input, OutputRange& output,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::test_forward<std::decay_t<Layer>>::call(
				l, batch_size, input, output, queue);
		}

		//
		// forward
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class forward {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(Layer& l,
						InputRange const& input, OutputRange& output,
						boost::compute::command_queue& queue) {
					l.forward(input, output, queue);
				}
			};
		}
		template<typename Layer, typename InputRange, typename OutputRange>
		decltype(auto) forward(Layer& l,
				InputRange const& input, OutputRange& output,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::forward<std::decay_t<Layer>>::call(
				l, input, output, queue);
		}

		//
		// backward_top
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class backward_top {
			public:
				template<typename InputRange>
				static decltype(auto) call(Layer& l,
						InputRange const& delta,
						boost::compute::command_queue& queue) {
					l.backward_top(delta, queue);
				}
			};
		}
		template<typename Layer, typename InputRange>
		decltype(auto) backward_top(Layer& l,
				InputRange const& delta,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::backward_top<std::decay_t<Layer>>::call(
				l, delta, queue);
		}

		//
		// backward
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class backward {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(Layer& l,
						InputRange const& delta, OutputRange& prev_delta,
						boost::compute::command_queue& queue) {
					l.backward(delta, prev_delta, queue);
				}
			};
		}
		template<typename Layer, typename InputRange, typename OutputRange>
		decltype(auto) backward(Layer& l,
				InputRange const& delta, OutputRange& prev_delta,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::backward<std::decay_t<Layer>>::call(
				l, delta, prev_delta, queue);
		}

		//
		// update
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class update {
			public:
				static decltype(auto) call(Layer& l,
						boost::compute::command_queue& queue) {
					l.update(queue);
				}
			};
		}
		template<typename Layer>
		decltype(auto) update(Layer& l,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::update<std::decay_t<Layer>>::call(l, queue);
		}

		//
		// serialize
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class serialize {
			public:
				static decltype(auto) call(Layer const& l,
						YAML::Emitter& emitter,
						boost::compute::command_queue& queue) {
					l.serialize(emitter, queue);
				}
			};
		}
		template<typename Layer>
		decltype(auto) serialize(Layer const& l,
				YAML::Emitter& emitter,
				boost::compute::command_queue& queue) {
			::neu::layer::traits::serialize<std::decay_t<Layer>>::call(l, emitter, queue);
		}
	}
}
#endif //NEU_LAYER_TRAITS_HPP
