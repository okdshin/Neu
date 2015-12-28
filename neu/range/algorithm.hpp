#ifndef NEU_RANGE_ALGORITHM_HPP
#define NEU_RANGE_ALGORITHM_HPP
//20151031
#include <type_traits>
#include <boost/compute/algorithm.hpp>
#include <neu/assert.hpp>
#include <neu/range/traits.hpp>
namespace neu {
	namespace range {
		template<typename InputRange, typename OutputRange>
		decltype(auto) copy(InputRange const& input, OutputRange& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range::distance(input) <= neu::range::distance(output));
			boost::compute::copy(neu::range::begin(input), neu::range::end(input),
				neu::range::begin(output), queue);
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) copy_async(InputRange const& input, OutputRange& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range::distance(input) <= neu::range::distance(output));
			return boost::compute::copy_async(neu::range::begin(input), neu::range::end(input),
				neu::range::begin(output), queue);
		}

		template<typename InputRange, typename OutputRange, typename Func>
		decltype(auto) transform(InputRange const& input, OutputRange& output,
				Func func,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range::distance(input) <= neu::range::distance(output));
			return boost::compute::transform(neu::range::begin(input), neu::range::end(input),
				neu::range::begin(output), func, queue);
		}
		template<typename InputRange1, typename InputRange2,
			typename OutputRange, typename Func,
			std::enable_if_t<!std::is_same<
				Func, boost::compute::command_queue>::value>* =nullptr>
		decltype(auto) transform(InputRange1 const& input1, InputRange2 const& input2,
				OutputRange& output, Func func,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(std::min(neu::range::distance(input1), neu::range::distance(input2))
				<= neu::range::distance(output));
			return boost::compute::transform(neu::range::begin(input1), neu::range::end(input1),
				neu::range::begin(input2), neu::range::begin(output), func, queue);
		}

		template<typename Range, typename T>
		decltype(auto) fill(Range const& range, T const& value,
				boost::compute::command_queue& queue) {
			return boost::compute::fill(
				neu::range::begin(range), neu::range::end(range), value, queue);
		}

		template<typename InputRange>
		decltype(auto) sum(InputRange const& input,
				boost::compute::command_queue& queue) {
			neu::scalar sum = 0.f;
			boost::compute::reduce(neu::range::begin(input), neu::range::end(input),
				&sum, queue);
			return sum;
		}

		template<typename InputRange1, typename InputRange2, typename OutputRange>
		decltype(auto) calc_last_layer_delta(
				InputRange1 const& last_output_range, InputRange2 const& teach_range,
				OutputRange& error_range,
				boost::compute::command_queue& queue) {
			return range::transform(last_output_range, teach_range, error_range,
				boost::compute::minus<neu::scalar>(), queue);
		}

		template<typename InputRange1, typename InputRange2>
		decltype(auto) mean_square_error(
				InputRange1 const& last_output_range, InputRange2 const& teach_range,
				boost::compute::command_queue& queue) {
			gpu_vector error(range::distance(last_output_range), queue.get_context());
			range::transform(last_output_range, teach_range, error,
				boost::compute::minus<neu::scalar>(), queue);
			static BOOST_COMPUTE_FUNCTION(float, square_kernel, (float x), {
				return x*x;
			});
			range::transform(error, error, square_kernel, queue);
			return range::sum(error, queue) / error.size();
		}

		template<typename InputRange1, typename InputRange2>
		decltype(auto) cross_entropy_loss(
				InputRange1 const& last_output_range, InputRange2 const& teach_range,
				boost::compute::command_queue& queue) {
			static BOOST_COMPUTE_FUNCTION(float, cross_entropy_kernel, (float d, float y), {
				return -d*log(y+0.00001);
			});
			gpu_vector error(range::distance(last_output_range), queue.get_context());
			range::transform(last_output_range, teach_range,
				error, cross_entropy_kernel, queue);
			return range::sum(error, queue);
		}

		/*
		decltype(auto) accuracy(std::size_t input_dim, std::size_t batch_size,
				gpu_vector const& last_output, gpu_vector const& teach) {
			NEU_ASSERT(teach.size() == last_output.size());
			NEU_ASSERT(input_dim*batch_size == last_output.size());
			scalar sum = 0.f;
			for(auto b = 0u; b < batch_size; ++b) {
				auto max_iter = boost::compute::
					max_element(last_output.begin()+b*input_dim,
						last_output.begin()+b*input_dim+input_dim);
				auto index = std::distance(max_iter, teach.begin()+b*input_dim);
				if(teach[index]) {
					sum += 1.f;
				}
			}
			return sum/batch_size;
		}
		*/
	}
}// namespace neu

#endif //NEU_RANGE_ALGORITHM_HPP
