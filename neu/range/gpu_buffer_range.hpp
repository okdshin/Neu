#ifndef NEU_RANGE_GPU_BUFFER_RANGE_HPP
#define NEU_RANGE_GPU_BUFFER_RANGE_HPP
//20151028
#include <memory>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
#include <neu/range/traits.hpp>
namespace neu {
	namespace range {
		template<typename T>
		class gpu_buffer_range {
		public:
			gpu_buffer_range() = default;
			gpu_buffer_range(
				boost::compute::buffer_iterator<T> begin,
				boost::compute::buffer_iterator<T> end)
			: begin_(begin), end_(end) {
				NEU_ASSERT(begin.get_index() <= end.get_index());
			}
			gpu_buffer_range(
				boost::compute::buffer_iterator<T> begin,
				std::size_t count)
			: gpu_buffer_range(begin, begin+count) {}

			decltype(auto) begin() const { return (begin_); }
			decltype(auto) end() const { return (end_); }
		private:
			boost::compute::buffer_iterator<T> begin_, end_;
		};
		using gpu_vector_range = gpu_buffer_range<scalar>;

		decltype(auto) to_range(gpu_vector const& v) {
			return gpu_vector_range(v.begin(), v.end());
		}
		template<typename T>
		decltype(auto) to_range(gpu_buffer_range<T> const& range) {
			return range;
		}
	}
}// namespace neu
namespace neu {
	namespace range {
		namespace traits {
			template<typename T>
			class get_buffer<neu::range::gpu_buffer_range<T>> {
			public:
				static decltype(auto) call(neu::range::gpu_buffer_range<T> const& range) {
					return range.begin().get_buffer();
				}
			};
		}
		namespace traits {
			template<typename T>
			class get_begin_index<neu::range::gpu_buffer_range<T>> {
			public:
				static decltype(auto) call(neu::range::gpu_buffer_range<T> const& range) {
					return range.begin().get_index();
				}
			};
		}
	}
}

#endif //NEU_RANGE_GPU_BUFFER_RANGE_HPP
