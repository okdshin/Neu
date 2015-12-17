#ifndef NEU_ACTIVATION_FUNC_DERIVATIVE_FOR_LOSS_HPP
#define NEU_ACTIVATION_FUNC_DERIVATIVE_FOR_LOSS_HPP
//20151031
#include <neu/basic_type.hpp>
#include <neu/range/algorithm.hpp>
namespace neu {
	class derivative_for_loss {
	public:
		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			return range::fill(output, 1.f, queue);
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_DERIVATIVE_FOR_LOSS_HPP
