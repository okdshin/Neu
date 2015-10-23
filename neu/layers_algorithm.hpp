#ifndef NEU_LAYERS_ALGORITHM_HPP
#define NEU_LAYERS_ALGORITHM_HPP
//20150914
#include <boost/timer.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	template<typename Layers>
	decltype(auto) layers_forward(Layers& layers, gpu_vector input) {
		using std::begin;
		using std::end;
		auto i = 0u;
		boost::timer timer;
		for(auto first = begin(layers); first != end(layers); ++first) {
			std::cout << "forward layer " << i << "------" << std::endl;
			timer.restart();
			first->forward(input);
			std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			input = first->get_next_input();
			++i;
		}
	}

	template<typename Layers>
	decltype(auto) layers_backward(Layers& layers, gpu_vector delta) {
		using std::rbegin;
		using std::rend;
		auto i = layers.size()-1;
		boost::timer timer;
		for(auto first = rbegin(layers); first != rend(layers); ++first) {
			timer.restart();
			std::cout << "backward layer " << i << "------" << std::endl;
			first->backward(delta);
			std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			delta = first->get_prev_delta();
			--i;
		}
	}

	template<typename Layers>
	decltype(auto) layers_update(Layers& layers) {
		using std::begin;
		using std::end;
		auto i = 0u;
		boost::timer timer;
		for(auto first = begin(layers); first != end(layers); ++first) {
			std::cout << "update layer " << i << "------" << std::endl;
			timer.restart();
			first->update();
			std::cout << "this layer consumed: " << timer.elapsed() << " secs" << std::endl;
			++i;
		}
	}

	decltype(auto) calc_last_layer_delta(
			gpu_vector const& last_output, gpu_vector const& teach) {
		gpu_vector error(last_output.size());
		boost::compute::transform(last_output.begin(), last_output.end(),
			teach.begin(), error.begin(), boost::compute::minus<neu::scalar>());
		return error;
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

	decltype(auto) mean_square_error(gpu_vector error) {
		boost::compute::transform(error.begin(), error.end(),
			error.begin(), error.begin(), boost::compute::multiplies<scalar>());
		scalar sum = 0.f;
		boost::compute::reduce(error.begin(), error.end(), &sum);
		return sum/error.size();
	}

	decltype(auto) cross_entropy_loss(
			gpu_vector const& last_output, gpu_vector const& teach) {
		static BOOST_COMPUTE_FUNCTION(float, cross_entropy_kernel, (float d, float y), {
			return -d*log(y);
		});
		gpu_vector cross_entropy(last_output.size());
		boost::compute::transform(teach.begin(), teach.end(),
			last_output.begin(), cross_entropy.begin(), cross_entropy_kernel);
		scalar sum = 0.f;
		boost::compute::reduce(cross_entropy.begin(), cross_entropy.end(), &sum);
		return sum;
	}
}// namespace neu

#endif //NEU_LAYERS_ALGORITHM_HPP
