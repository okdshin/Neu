#include <iostream>
#include <neu/layer.hpp>

decltype(auto) call_layer_member_function(neu::layer& l) {
	neu::gpu_vector input, delta;
	l.test_forward(input);
	l.forward(input);
	l.get_next_input();
	l.backward(delta);
	l.get_prev_delta();
	l.update();
}

class MemFunLayer {
public:
	decltype(auto) test_forward(neu::gpu_vector) {
		std::cout << "test_forward" << std::endl;
	}
	decltype(auto) forward(neu::gpu_vector) {
		std::cout << "forward" << std::endl;
	}
	decltype(auto) get_next_input() const {
		std::cout << "get_next_input" << std::endl;
		return (next_input_);
	}

	decltype(auto) backward(neu::gpu_vector) {
		std::cout << "backward" << std::endl;
	}
	decltype(auto) get_prev_delta() const {
		std::cout << "get_prev_delta" << std::endl;
		return (prev_delta_);
	}

	decltype(auto) should_update() const {
		std::cout << "should update" << std::endl;
		return true;
	}
	decltype(auto) update() {
		std::cout << "update" << std::endl;
	}

private:
	neu::gpu_vector next_input_, prev_delta_;

};

class TraitsLayer {
public:
	neu::gpu_vector next_input, prev_delta;
};

namespace neu_layer_traits {
	template<>
	class layer_test_forward<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer& l, neu::gpu_vector const& input) {
			std::cout << "test_forward" << std::endl;
		}
	};
	template<>
	class layer_forward<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer& l, neu::gpu_vector const& input) {
			std::cout << "forward" << std::endl;
		}
	};
	template<>
	class layer_get_next_input<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer const& l) {
			std::cout << "get_next_input" << std::endl;
			return (l.next_input);
		}
	};
	template<>
	class layer_backward<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer& l, neu::gpu_vector const& delta) {
			std::cout << "backward" << std::endl;
		}
	};
	template<>
	class layer_get_prev_delta<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer const& l) {
			std::cout << "get_prev_delta" << std::endl;
			return (l.prev_delta);
		}
	};
	template<>
	class layer_should_update<TraitsLayer> {
	public:
		static bool call(TraitsLayer const& l) {
			std::cout << "should update" << std::endl;
			return false;
		}
	};
	template<>
	class layer_should_update<TraitsLayer> {
	public:
		static bool call(TraitsLayer const& l) {
			std::cout << "should update" << std::endl;
			return true;
		}
	};
	template<>
	class layer_update<TraitsLayer> {
	public:
		static decltype(auto) call(TraitsLayer& l) {
			std::cout << "update" << std::endl;
		}
	};
}

int main() {
	{
		std::cout << "mem_fun_layer:" << std::endl;
		neu::layer mem_fun_layer = MemFunLayer();
		call_layer_member_function(mem_fun_layer);
	}
	std::cout << "\n";
	{
		std::cout << "traits_layer:" << std::endl;
		neu::layer traits_layer = TraitsLayer();
		call_layer_member_function(traits_layer);
	}
}
