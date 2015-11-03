#ifndef NEU_LAYERS_ALGORITHM_HPP
#define NEU_LAYERS_ALGORITHM_HPP
//20150914
#include <boost/timer.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/gpu_buffer_range.hpp>
#include <neu/layer_traits.hpp>
#include <neu/layer.hpp>
namespace neu {
	decltype(auto) max_inoutput_size(std::vector<layer> const& layers) {
		auto max_in = std::max_element(layers.begin(), layers.end(),
			[](auto const& l, auto const& r) { return l.input_dim() < r.input_dim(); });
		auto max_out = std::max_element(layers.begin(), layers.end(),
			[](auto const& l, auto const& r) {
				return l.output_dim() < r.output_dim(); });
		auto max_batch = std::max_element(layers.begin(), layers.end(),
			[](auto const& l, auto const& r) {
				return l.batch_size() < r.batch_size(); });
		std::cout << max_in->input_dim() << std::endl;
		std::cout << max_in->output_dim() << std::endl;
		return std::max(max_in->input_dim(), max_out->output_dim())
			*max_batch->batch_size();
	}
	decltype(auto) layers_forward(std::vector<layer>& layers,
			gpu_vector_range const& input,
			gpu_vector& input_buffer, gpu_vector& output_buffer) {
		neu::range_copy(input, input_buffer);
		auto i = 0u;
		boost::timer timer;
		for(auto& l : layers) {
			neu::gpu_vector_range input_range(input_buffer.begin(),
				neu::layer_input_dim(l)*neu::layer_batch_size(l));
			neu::gpu_vector_range output_range(output_buffer.begin(),
				neu::layer_output_dim(l)*neu::layer_batch_size(l));
			std::cout << "forward layer " << i << "------" << std::endl;
			timer.restart();
			l.forward(input_range, output_range);
			std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			input_buffer.swap(output_buffer);
			++i;
		}
		neu::gpu_vector_range output_range(output_buffer.begin(),
			neu::layer_output_dim(layers.back())*neu::layer_batch_size(layers.back()));
		return output_range;
	}

	decltype(auto) layers_backward(std::vector<layer>& layers,
			gpu_vector_range const& delta,
			gpu_vector& delta_buffer, gpu_vector& prev_delta_buffer) {
		neu::range_copy(delta, delta_buffer);
		neu::gpu_vector_range delta_range(delta_buffer.begin(),
			neu::range_distance(delta));
		neu::gpu_vector_range prev_delta_range;
		auto i = layers.size()-1;
		boost::timer timer;
		for(auto first = layers.rbegin(); first != layers.rend(); ++first) {
			auto& l = *first;
			auto prev_delta_size = l.input_dim()*l.batch_size();
			prev_delta_range = neu::gpu_vector_range(prev_delta_buffer.begin(),
				prev_delta_size);
			std::cout << "backward layer " << i << "------" << std::endl;
			timer.restart();
			l.backward(delta_range, prev_delta_range);
			std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			delta_buffer.swap(prev_delta_buffer);
			delta_range = neu::gpu_vector_range(delta_buffer.begin(), prev_delta_size);
			--i;
		}
		return prev_delta_range;
	}

	decltype(auto) layers_update(std::vector<layer>& layers) {
		auto i = 0u;
		boost::timer timer;
		for(auto& l : layers) {
			if(l.should_update()) {
				std::cout << "update layer " << i << "------" << std::endl;
				timer.restart();
				l.update();
				std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			}	
			++i;
		}
	}

	template<typename InputRange1, typename InputRange2, typename OutputRange>
	decltype(auto) calc_last_layer_delta(
			InputRange1 const& last_output_range, InputRange2 const& teach_range,
			OutputRange const& error_range) {
		return neu::range_transform(last_output_range, teach_range, error_range,
			boost::compute::minus<neu::scalar>());
	}

	decltype(auto) mean_square_error(
			gpu_vector const& last_output, gpu_vector const& teach) {
		gpu_vector error(last_output.size());
		boost::compute::transform(last_output.begin(), last_output.end(),
			teach.begin(), error.begin(), boost::compute::minus<neu::scalar>());
		boost::compute::transform(error.begin(), error.end(),
			error.begin(), error.begin(), boost::compute::multiplies<scalar>());
		scalar sum = 0.f;
		boost::compute::reduce(error.begin(), error.end(), &sum);
		return sum/error.size();
	}

	template<typename Range>
	decltype(auto) mean_square_error(Range const& error) {
		neu::range_transform(error, error, error, boost::compute::multiplies<scalar>());
		scalar sum = 0.f;
		boost::compute::reduce(neu::range_begin(error), neu::range_end(error), &sum);
		return sum/neu::range_distance(error);
	}

	template<typename Range1, typename Range2>
	decltype(auto) cross_entropy_loss(
			Range1 const& last_output, Range2 const& teach, gpu_vector& buffer) {
		static BOOST_COMPUTE_FUNCTION(float, cross_entropy_kernel, (float d, float y), {
			return -d*log(y+0.00001);
		});
		auto end = neu::range_transform(last_output, teach, buffer, cross_entropy_kernel);
		scalar sum = 0.f;
		boost::compute::reduce(buffer.begin(), end, &sum);
		return sum;
	}

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
}// namespace neu

#endif //NEU_LAYERS_ALGORITHM_HPP
