#ifndef NEU_LAYERS_ALGORITHM_HPP
#define NEU_LAYERS_ALGORITHM_HPP
//20150914
#include <boost/timer.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	template<typename Layers>
	decltype(auto) layers_forward(Layers& layers, gpu_vector input) {
		using std::begin;
		using std::end;
		auto i = 0u;
		boost::timer timer;
		for(auto first = begin(layers); first != end(layers); ++first) {
			std::cout << i << "------" << std::endl;
			timer.restart();
			first->forward(input);
			std::cout << timer.elapsed() << std::endl;
			input = first->get_next_input();
			++i;
		}
	}
	template<typename Layers>
	decltype(auto) layers_backward(Layers& layers, gpu_vector delta) {
		using std::rbegin;
		using std::rend;
		auto i = layers.size();
		boost::timer timer;
		for(auto first = rbegin(layers); first != rend(layers); ++first) {
			timer.restart();
			std::cout << i << "------" << std::endl;
			first->backward(delta);
			std::cout << timer.elapsed() << std::endl;
			delta = first->get_prev_delta();
			--i;
		}
	}
}// namespace neu

#endif //NEU_LAYERS_ALGORITHM_HPP
