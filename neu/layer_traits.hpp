#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20151025
#include <type_traits>
#include <gsl.h>
//
// layer_test_forward
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class test_forward {
	public:
		static decltype(auto) call(Layer& l, neu::gpu_vector const& input) {
			l.test_forward(input);
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_test_forward(Layer& l, gpu_vector const& input) {
		neu_layer_traits::test_forward<std::decay_t<Layer>>::call(l, input);
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
		static decltype(auto) call(Layer& l, neu::gpu_vector const& input) {
			l.forward(input);
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_forward(Layer& l, gpu_vector const& input) {
		neu_layer_traits::forward<std::decay_t<Layer>>::call(l, input);
	}
}

//
// layer_get_next_input
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class get_next_input {
	public:
		static decltype(auto) call(Layer const& l) {
			return l.get_next_input();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_get_next_input(Layer const& l) {
		return neu_layer_traits::get_next_input<std::decay_t<Layer>>::call(l);
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
		static decltype(auto) call(Layer& l, neu::gpu_vector const& delta) {
			l.backward(delta);
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_backward(Layer& l, gpu_vector const& delta) {
		neu_layer_traits::backward<std::decay_t<Layer>>::call(l, delta);
	}
}

//
// layer_get_prev_delta
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class get_prev_delta {
	public:
		static decltype(auto) call(Layer const& l) {
			return l.get_prev_delta();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_get_prev_delta(Layer const& l) {
		return neu_layer_traits::get_prev_delta<std::decay_t<Layer>>::call(l);
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
			Expects(neu::layer_should_update(l));
			//l.update();
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
