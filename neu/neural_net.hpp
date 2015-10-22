#ifndef NEU_NEURAL_NET_HPP
#define NEU_NEURAL_NET_HPP
//20151022
#include <neu/layers_algorithm.hpp>
namespace neu {
	class neural_net {
	public:
		template<typename Layer>
		decltype(auto) add_layer(Layer const& layer) {
			layers_.push_back(layer);
		}

		decltype(auto) forward(gpu_vector const& input) {
			neu::layers_forward(layers_, input);
		}
		decltype(auto) get_next_input() const {
			return layers_.back().get_next_input();
		}

		decltype(auto) backward(gpu_vector const& delta) {
			neu::layers_backward(layers_, delta);
		}
		decltype(auto) get_prev_delta() const {
			return layers_.front().get_prev_delta();
		}

		decltype(auto) update() {
			neu::layers_update(layers_);
		}

	private:
		std::vector<neu::layer> layers_;
	};
	
}// namespace neu

#endif //NEU_NEURAL_NET_HPP
