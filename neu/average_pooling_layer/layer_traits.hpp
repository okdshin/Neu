#ifndef NEU_AVERAGE_POOLING_LAYER_LAYER_TRAITS_HPP
#define NEU_AVERAGE_POOLING_LAYER_LAYER_TRAITS_HPP
//20151026
#include <neu/layer_traits.hpp>
#include <neu/average_pooling_layer/impl.hpp>
namespace neu_layer_traits {
	template<>
	class forward<neu::average_pooling_layer> {
	public:
		template<typename InputRange, typename OutputRange>
		static decltype(auto) call(neu::average_pooling_layer& l,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			l.test_forward(l.batch_size(), input, output, queue);
		}
	};
	template<>
	class should_update<neu::average_pooling_layer> {
	public:
		static bool call(neu::average_pooling_layer const& l) {
			return false;
		}
	};
}
#endif //NEU_AVERAGE_POOLING_LAYER_LAYER_TRAITS_HPP
