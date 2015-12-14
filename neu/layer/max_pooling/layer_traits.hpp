#ifndef NEU_MAX_POOLING_LAYER_LAYER_TRAITS_HPP
#define NEU_MAX_POOLING_LAYER_LAYER_TRAITS_HPP
//20151026
#include <neu/layer_traits.hpp>
#include <neu/max_pooling_layer/impl.hpp>
namespace neu_layer_traits {
	template<>
	class forward<neu::max_pooling_layer> {
	public:
		template<typename InputRange, typename OutputRange>
		static decltype(auto) call(neu::max_pooling_layer& l,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			l.test_forward(l.batch_size(), input, output, queue);
		}
	};
	template<>
	class should_update<neu::max_pooling_layer> {
	public:
		static bool call(neu::max_pooling_layer const& l) {
			return false;
		}
	};
}
#endif //NEU_MAX_POOLING_LAYER_LAYER_TRAITS_HPP
