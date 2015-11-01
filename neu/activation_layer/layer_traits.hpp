#ifndef NEU_ACTIVATION_LAYER_LAYER_TRAITS_HPP
#define NEU_ACTIVATION_LAYER_LAYER_TRAITS_HPP
//20151025
#include <neu/layer_traits.hpp>
#include <neu/activation_layer/impl.hpp>
namespace neu_layer_traits {
	template<typename ActivationFunc, typename DiffActivationFunc>
	class should_update<neu::activation_layer<ActivationFunc, DiffActivationFunc>> {
	public:
		static decltype(auto) call(
				neu::activation_layer<ActivationFunc, DiffActivationFunc> const& l) {
			return false;
		}
	};
}

#endif //NEU_ACTIVATION_LAYER_LAYER_TRAITS_HPP
