#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20151025
#include <type_traits>
//
// layer_input_dim
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class input_dim {
	public:
		static decltype(auto) call(Layer const& l) {
			return l.input_dim();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_input_dim(Layer const& l) {
		return neu_layer_traits::input_dim<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_output_dim
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class output_dim {
	public:
		static decltype(auto) call(Layer const& l) {
			return l.output_dim();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_output_dim(Layer const& l) {
		return neu_layer_traits::output_dim<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_batch_size
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class batch_size {
	public:
		static decltype(auto) call(Layer const& l) {
			return l.batch_size();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_batch_size(Layer const& l) {
		return neu_layer_traits::batch_size<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_test_forward
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class test_forward {
	public:
		template<typename InputRange, typename OutputRange>
		static decltype(auto) call(Layer& l, std::size_t batch_size,
				InputRange const& input, OutputRange const& output) {
			l.test_forward(batch_size, input, output);
		}
	};
}
namespace neu {
	template<typename Layer, typename InputRange, typename OutputRange>
	decltype(auto) layer_test_forward(Layer& l, std::size_t batch_size,
			InputRange const& input, OutputRange const& output) {
		neu_layer_traits::test_forward<std::decay_t<Layer>>::
			call(l, batch_size, input, output);
	}
}

//
// layer_forward
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class forward {
	public:
		template<typename InputRange, typename OutputRange>
		static decltype(auto) call(Layer& l,
				InputRange const& input, OutputRange const& output) {
			l.forward(input, output);
		}
	};
}
namespace neu {
	template<typename Layer, typename InputRange, typename OutputRange>
	decltype(auto) layer_forward(Layer& l,
			InputRange const& input, OutputRange const& output) {
		neu_layer_traits::forward<std::decay_t<Layer>>::call(l, input, output);
	}
}

//
// layer_backward
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class backward {
	public:
		template<typename InputRange, typename OutputRange>
		static decltype(auto) call(Layer& l,
				InputRange const& delta, OutputRange const& prev_delta) {
			l.backward(delta, prev_delta);
		}
	};
}
namespace neu {
	template<typename Layer, typename InputRange, typename OutputRange>
	decltype(auto) layer_backward(Layer& l,
			InputRange const& delta, OutputRange const& prev_delta) {
		neu_layer_traits::backward<std::decay_t<Layer>>::call(l, delta, prev_delta);
	}
}

//
// layer_should_update
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class should_update {
	public:
		static bool call(Layer const& l) {
			return l.should_update();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_should_update(Layer const& l) {
		return neu_layer_traits::should_update<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_update
//
namespace neu_layer_traits {
	// default implementation (do nothing)
	template<typename Layer>
	class update {
	public:
		static decltype(auto) call(Layer& l) {
			assert(!"you should specialize neu_layer_traits::update"); //TODO
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_update(Layer& l) {
		neu_layer_traits::update<std::decay_t<Layer>>::call(l);
	}
}
#endif //NEU_LAYER_TRAITS_HPP
