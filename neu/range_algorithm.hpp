#ifndef NEU_RANGE_ALGORITHM_HPP
#define NEU_RANGE_ALGORITHM_HPP
//20151031
#include <type_traits>
#include <boost/compute/algorithm.hpp>
#include <neu/range_traits.hpp>
namespace neu {
	template<typename InputRange, typename OutputRange>
	decltype(auto) range_copy(InputRange const& input, OutputRange& output,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		NEU_ASSERT(neu::range_distance(input) <= neu::range_distance(output));
		boost::compute::copy(neu::range_begin(input), neu::range_end(input),
			neu::range_begin(output), queue);
	}

	template<typename InputRange, typename OutputRange>
	decltype(auto) range_copy_async(InputRange const& input, OutputRange& output,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		NEU_ASSERT(neu::range_distance(input) <= neu::range_distance(output));
		return boost::compute::copy_async(neu::range_begin(input), neu::range_end(input),
			neu::range_begin(output), queue);
	}

	template<typename InputRange, typename OutputRange, typename Func>
	decltype(auto) range_transform(InputRange const& input, OutputRange& output,
			Func func,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		NEU_ASSERT(neu::range_distance(input) <= neu::range_distance(output));
		return boost::compute::transform(neu::range_begin(input), neu::range_end(input),
			neu::range_begin(output), func, queue);
	}
	template<typename InputRange1, typename InputRange2,
		typename OutputRange, typename Func,
		std::enable_if_t<!std::is_same<
			Func, boost::compute::command_queue>::value>* =nullptr>
	decltype(auto) range_transform(InputRange1 const& input1, InputRange2 const& input2,
			OutputRange& output, Func func,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		NEU_ASSERT(std::min(neu::range_distance(input1), neu::range_distance(input2))
			<= neu::range_distance(output));
		return boost::compute::transform(neu::range_begin(input1), neu::range_end(input1),
			neu::range_begin(input2), neu::range_begin(output), func, queue);
	}

	template<typename Range, typename T>
	decltype(auto) range_fill(Range const& range, T const& value,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		return boost::compute::fill(
			neu::range_begin(range), neu::range_end(range), value, queue);
	}

	template<typename InputRange>
	decltype(auto) range_sum(InputRange const& input,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		neu::scalar sum = 0.f;
		boost::compute::reduce(neu::range_begin(input), neu::range_end(input),
			&sum, queue);
		return sum;
	}
}// namespace neu

#endif //NEU_RANGE_ALGORITHM_HPP
