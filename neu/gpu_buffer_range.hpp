#ifndef NEU_GPU_BUFFER_RANGE_HPP
#define NEU_GPU_BUFFER_RANGE_HPP
//20151028
#include <memory>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_traits.hpp>
namespace neu {
	template<typename T>
	class gpu_buffer_range {
	public:
		gpu_buffer_range(
			boost::compute::buffer_iterator<T> begin,
			boost::compute::buffer_iterator<T> end)
		: buffer_(begin.get_buffer()), begin_(begin), end_(end) {
			NEU_ASSERT(begin.get_index() <= end.get_index());
		}
		decltype(auto) begin() const { return (begin_); }
		decltype(auto) end() const { return (end_); }
	private:
		boost::compute::buffer buffer_;
		boost::compute::buffer_iterator<T> begin_, end_;
	};
	using gpu_vector_range = gpu_buffer_range<scalar>;
	decltype(auto) to_range(gpu_vector const& v) {
		return gpu_vector_range(v.begin(), v.end());
	}
}// namespace neu
namespace neu_range_traits {
	template<typename T>
	class size<neu::gpu_buffer_range<T>> {
	public:
		static decltype(auto) call(neu::gpu_buffer_range<T> const& range) {
			return range.end().get_index()-range.begin().get_index();
		}
	};
	template<typename T>
	class get_buffer<neu::gpu_buffer_range<T>> {
	public:
		static decltype(auto) call(neu::gpu_buffer_range<T> const& range) {
			return range.begin().get_buffer();
		}
	};
}

#endif //NEU_GPU_BUFFER_RANGE_HPP
