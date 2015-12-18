#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20151025
#include <type_traits>
#include <yaml-cpp/yaml.h>

namespace neu {
	namespace layer {
		//
		// input_dim
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class input_dim {
			public:
				static decltype(auto) call(Layer const& l) {
					return l.input_dim();
				}
			};
		}
		template<typename Layer>
		decltype(auto) input_dim(Layer const& l) {
			return ::neu::layer::traits::input_dim<std::decay_t<Layer>>::call(l);
		}

		//
		// output_dim
		//
		namespace traits {
			// default implementation (call member function)
			template<typename Layer>
			class output_dim {
			public:
				static decltype(auto) call(Layer const& l) {
					return l.output_dim();
				}
			};
		}
		template<typename Layer>
		decltype(auto) output_dim(Layer const& l) {
			return ::neu::layer::traits::output_dim<std::decay_t<Layer>>::call(l);
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
		decltype(auto) input_size(Layer const& l) {
			return ::neu::layer::input_dim(l)*::neu::layer::batch_size(l);
		}

		// output_size
		template<typename Layer>
		decltype(auto) output_size(Layer const& l) {
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
