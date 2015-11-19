#ifndef NEU_LAYERS_ALGORITHM_HPP
#define NEU_LAYERS_ALGORITHM_HPP
//20150914
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
		return std::max(max_in->input_dim(), max_out->output_dim())
			*max_batch->batch_size();
	}

	decltype(auto) layers_test_forward(std::vector<layer>& layers,
			std::size_t batch_size,
			gpu_vector const& initial_input, gpu_vector& output,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		gpu_vector input(initial_input.begin(), initial_input.end(), queue);
		for(auto& l : layers) {
			output.resize(neu::layer_output_size(l), queue);
			l.test_forward(batch_size, to_range(input), to_range(output), queue);
			input.swap(output);
		}
	}

	decltype(auto) layers_forward(std::vector<layer>& layers,
			gpu_vector const& initial_input, gpu_vector& output,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		gpu_vector input(initial_input.begin(), initial_input.end(), queue);
		for(auto& l : layers) {
			output.resize(neu::layer_output_size(l), queue);
			l.forward(to_range(input), to_range(output), queue);
			input.swap(output);
		}
		input.swap(output);
	}

	decltype(auto) layers_backward(std::vector<layer>& layers,
			gpu_vector const& initial_delta, gpu_vector& prev_delta,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		gpu_vector delta(initial_delta.begin(), initial_delta.end(), queue);
		for(int i = layers.size()-1; i >= 0; --i) {
			auto& l = layers.at(i);
			prev_delta.resize(neu::layer_input_size(l), queue);
			l.backward(to_range(delta), to_range(prev_delta), queue);
			delta.swap(prev_delta);
		}
		delta.swap(prev_delta);
	}

	// TODO update async
	decltype(auto) layers_update(std::vector<layer>& layers,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		for(auto& l : layers) {
			if(l.should_update()) {
				l.update(queue);
			}	
		}
	}

	template<typename InputRange1, typename InputRange2, typename OutputRange>
	decltype(auto) calc_last_layer_delta(
			InputRange1 const& last_output_range, InputRange2 const& teach_range,
			OutputRange& error_range,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		return neu::range_transform(last_output_range, teach_range, error_range,
			boost::compute::minus<neu::scalar>(), queue);
	}

	template<typename InputRange1, typename InputRange2>
	decltype(auto) mean_square_error(
			InputRange1 const& last_output_range, InputRange2 const& teach_range,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		gpu_vector error(neu::range_distance(last_output_range), queue.get_context());
		neu::range_transform(last_output_range, teach_range, error,
			boost::compute::minus<neu::scalar>(), queue);
		static BOOST_COMPUTE_FUNCTION(float, square_kernel, (float x), {
			return x*x;
		});
		neu::range_transform(error, error, square_kernel, queue);
		return neu::range_sum(error, queue) / error.size();
	}

	template<typename InputRange1, typename InputRange2>
	decltype(auto) cross_entropy_loss(
			InputRange1 const& last_output_range, InputRange2 const& teach_range,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		static BOOST_COMPUTE_FUNCTION(float, cross_entropy_kernel, (float d, float y), {
			return -d*log(y+0.00001);
		});
		gpu_vector error(neu::range_distance(last_output_range), queue.get_context());
		neu::range_transform(last_output_range, teach_range,
			error, cross_entropy_kernel, queue);
		return neu::range_sum(error, queue);
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
}// namespace neu

#endif //NEU_LAYERS_ALGORITHM_HPP
