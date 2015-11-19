#ifndef NEU_CONVOLUTION_LAYER_LAYER_TRAITS_HPP
#define NEU_CONVOLUTION_LAYER_LAYER_TRAITS_HPP
//20151025
#include <neu/layer_traits.hpp>
#include <neu/convolution_layer/impl.hpp>
namespace neu_layer_traits {
	template<typename LearningRateGen>
	class should_update<neu::convolution_layer<LearningRateGen>> {
	public:
		static bool call(neu::convolution_layer<LearningRateGen> const& l) {
			return true;
		}
	};
	template<typename LearningRateGen>
	class update<neu::convolution_layer<LearningRateGen>> {
	public:
		static decltype(auto) call(neu::convolution_layer<LearningRateGen>& l,
			boost::compute::command_queue& queue) {
			l.update(queue);
		}
	};
}

#endif //NEU_CONVOLUTION_LAYER_LAYER_TRAITS_HPP
