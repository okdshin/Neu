#ifndef NEU_LEARNING_RATE_GEN_HPP
#define NEU_LEARNING_RATE_GEN_HPP
//20151023
#include <functional>
namespace neu {
	using learning_rate_gen =
		std::function<void(gpu_vector&, gpu_vector&, gpu_vector const&, gpu_vector const&)>;
}// namespace neu

#endif //NEU_LEARNING_RATE_GEN_HPP
