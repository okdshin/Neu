#ifndef NEU_FULLY_CONNECTED_LAYER_LAYER_TRAITS_HPP
#define NEU_FULLY_CONNECTED_LAYER_LAYER_TRAITS_HPP
//20151025
#include <neu/layer_traits.hpp>
#include <neu/fully_connected_layer/impl.hpp>
namespace neu_layer_traits {
	template<typename LearningRateGen>
	class should_update<neu::fully_connected_layer<LearningRateGen>> {
	public:
		static bool call(neu::fully_connected_layer<LearningRateGen> const& l) {
			return true;
		}
	};
	template<typename LearningRateGen>
	class update<neu::fully_connected_layer<LearningRateGen>> {
	public:
		static decltype(auto) call(neu::fully_connected_layer<LearningRateGen>& l,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			l.update(queue);
		}
	};
}

#endif //NEU_FULLY_CONNECTED_LAYER_LAYER_TRAITS_HPP
