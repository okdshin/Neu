#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20151025
#include <type_traits>
#include <boost/tti/has_type.hpp>

namespace neu {
	class layer_tag {};
	class fully_connected_like_layer_tag : public layer_tag {};
	class convolution_like_layer_tag : public layer_tag {};
}
namespace neu_layer_traits {
	BOOST_TTI_TRAIT_HAS_TYPE(layer_has_layer_category, layer_category);
	
	template<typename, bool>
	class category_impl {};
	template<typename Layer>
	class category_impl<Layer, true> {
	public:
		using type = typename Layer::layer_category;
	};
	template<typename Layer>
	class category_impl<Layer, false> {
	public:
		using type = neu::layer_tag;
	};

	template<typename Layer>
	class category {
	public:
		using type = typename neu_layer_traits::category_impl<Layer,
			neu_layer_traits::layer_has_layer_category<Layer>::value>::type;
	};
}
namespace neu {
	template<typename Layer>
	using layer_category_t = typename neu_layer_traits::category<Layer>::type;
}

//
// layer_input_width
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class input_width {
	public:
		static decltype(auto) call(Layer const& l) {
			static_assert(std::is_same<
				neu::layer_category_t<Layer>, neu::convolution_like_layer_tag>::value, "");
			return l.input_width();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_input_width(Layer const& l) {
		return neu_layer_traits::input_width<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_input_channel_num
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class input_channel_num {
	public:
		static decltype(auto) call(Layer const& l) {
			static_assert(std::is_same<
				neu::layer_category_t<Layer>, neu::convolution_like_layer_tag>::value, "");
			return l.input_channel_num();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_input_channel_num(Layer const& l) {
		return neu_layer_traits::input_channel_num<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_input_dim
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class input_dim {
	public:
		static decltype(auto) call(Layer const& l) {
			return call_impl(l, neu::layer_category_t<Layer>());
		}
	private:
		static decltype(auto) call_impl(Layer const& l, neu::convolution_like_layer_tag) {
			return neu::layer_input_width(l)*neu::layer_input_width(l)
				*neu::layer_input_channel_num(l);
		}
		static decltype(auto) call_impl(Layer const& l, neu::layer_tag) {
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
// layer_output_width
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class output_width {
	public:
		static decltype(auto) call(Layer const& l) {
			static_assert(std::is_same<
				neu::layer_category_t<Layer>, neu::convolution_like_layer_tag>::value, "");
			return l.output_width();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_output_width(Layer const& l) {
		return neu_layer_traits::output_width<std::decay_t<Layer>>::call(l);
	}
}

//
// layer_output_channel_num
//
namespace neu_layer_traits {
	// default implementation (call member function)
	template<typename Layer>
	class output_channel_num {
	public:
		static decltype(auto) call(Layer const& l) {
			static_assert(std::is_same<
				neu::layer_category_t<Layer>, neu::convolution_like_layer_tag>::value, "");
			return l.output_channel_num();
		}
	};
}
namespace neu {
	template<typename Layer>
	decltype(auto) layer_output_channel_num(Layer const& l) {
		return neu_layer_traits::output_channel_num<std::decay_t<Layer>>::call(l);
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
			return call_impl(l, neu::layer_category_t<Layer>());
		}
	private:
		static decltype(auto) call_impl(Layer const& l, neu::convolution_like_layer_tag) {
			return neu::layer_output_width(l)*neu::layer_output_width(l)
				*neu::layer_output_channel_num(l);
		}
		static decltype(auto) call_impl(Layer const& l, neu::layer_tag) {
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
