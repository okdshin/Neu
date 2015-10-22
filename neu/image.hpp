#ifndef NEU_IMAGE_HPP
#define NEU_IMAGE_HPP
//20150611
#include <boost/filesystem/path.hpp>
#include <FreeImagePlus.h>
#include <neu/basic_type.hpp>
namespace neu {
decltype(auto) load_rgb_image_as_3ch_image_vector(std::string const& filename) {
	fipImage image;
	if(!image.load(filename.c_str())) {
		throw "image load error";
	}
	image.convertTo32Bits();
	std::vector<neu::scalar> cpu_vec(3*image.getHeight()*image.getWidth());
	for(auto y = 0u; y < image.getHeight(); ++y) {
		BYTE* row = image.getScanLine(y);
		for(auto x = 0u; x < image.getWidth(); ++x) {
			assert(y*image.getWidth()+x < cpu_vec.size());
			assert(image.getHeight()*image.getWidth()+y*image.getWidth()+x < cpu_vec.size());
			assert(2*image.getHeight()*image.getWidth()+y*image.getWidth()+x < cpu_vec.size());
			// coordinate system of freeimage is upside down.
			auto oy = (static_cast<int>(image.getHeight())-1) - y;
			cpu_vec[oy*image.getWidth()+x] = *(row+4*x)/255.0;
			cpu_vec[image.getHeight()*image.getWidth()+oy*image.getWidth()+x] =
				*(row+4*x+1)/255.0;
			cpu_vec[2*image.getHeight()*image.getWidth()+oy*image.getWidth()+x] =
				*(row+4*x+2)/255.0;
		}
	}
	return std::make_tuple(cpu_vec, image.getWidth(), image.getHeight());
}

template<typename InIter, typename OutIter>
decltype(auto) normalize(InIter first, InIter const& last, OutIter out) {
	auto max = *std::max_element(first, last);
	auto sum = 0.0;
	for(auto iter = first; iter != last; ++iter) {
		sum += *iter;
	}
	auto mean = sum/(last-first);
	for(auto iter = first; iter != last; ++iter) {
		auto e = (*iter-mean)/(max-mean);
		e = e < 0 ? 0 : e;
		*out = e;
		++out;
	}
	/*
	auto min = *std::min_element(first, last);
	for(auto iter = first; iter != last; ++iter) {
		e = (e-min)/(max-min);
		e = e < 0 ? 0 : e;
	}
	*/
}

template<typename Iter>
decltype(auto) save_image_vector_as_image(Iter first, Iter const& last,
		std::size_t width, boost::filesystem::path const& filepath, scalar ratio) {
	auto image = fipImage{FIT_BITMAP,
		static_cast<unsigned int>(width), static_cast<unsigned int>(width), 32};
	// coordinate system of freeimage is upside down.
	for(auto y = static_cast<int>(width)-1; y >= 0; --y) {
		BYTE* row = image.getScanLine(y);
		for(auto x = 0u; x < width; ++x) {
			assert(first != last);
			*(row+x*4) = *(row+x*4+1) = *(row+x*4+2) = ratio*(*first);
			++first;
		}
	}
	image.save(filepath.string().c_str());
	assert(first == last);
}

template<typename Iter>
decltype(auto) save_image_vector_as_images(
		Iter first, Iter const& last,
		std::size_t width, std::size_t channel_num, std::size_t batch_size,
		boost::filesystem::path const& filepath, scalar ratio) {
	for(auto b = 0u; b < batch_size; ++b) {
		for(auto k = 0u; k < channel_num; ++k) {
			assert(first != last);
			auto next_first = first+width*width;
			neu::cpu_vector normalized(width*width);
			neu::normalize(first, next_first, normalized.begin());
			neu::save_image_vector_as_image(normalized.begin(), normalized.end(),
				width, filepath.parent_path().string()+filepath.stem().string()+std::to_string(b)+std::to_string(k)+filepath.extension().string(), ratio);
			first = next_first;
		}
	}
	assert(first == last);
}
decltype(auto) save_image_vector_as_images(
		neu::cpu_vector const& image_vector,
		std::size_t width, std::size_t channel_num, std::size_t batch_size,
		boost::filesystem::path const& filepath, scalar ratio) {
	neu::save_image_vector_as_images(image_vector.begin(), image_vector.end(),
		width, channel_num, batch_size, filepath, ratio);
}

template<typename Iter>
decltype(auto) save_3ch_image_vector_as_rgb_image(Iter first, Iter const& last,
		std::size_t width, boost::filesystem::path const& filepath, scalar ratio) {
	auto image = fipImage{FIT_BITMAP,
		static_cast<unsigned int>(width), static_cast<unsigned int>(width), 32};
	// coordinate system of freeimage is upside down.
	for(auto y = static_cast<int>(width)-1; y >= 0; --y) {
		BYTE* row = image.getScanLine(y);
		for(auto x = 0u; x < width; ++x) {
			assert(first != last);
			*(row+x*4) = ratio*(*(first+2*width*width));
			*(row+x*4+1) = ratio*(*(first+width*width));
			*(row+x*4+2) = ratio*(*first);
			++first;
		}
	}
	image.save(filepath.string().c_str());
	assert(first == last);
}
decltype(auto) save_3ch_image_vector_as_rgb_image(
		neu::cpu_vector const& image_vector,
		std::size_t width, boost::filesystem::path const& filepath, scalar ratio) {
	neu::save_3ch_image_vector_as_rgb_image(image_vector.begin(), image_vector.end(),
		width, filepath, ratio);
}

template<typename Iter>
decltype(auto) save_1ch_image_vector_as_monochro_image(Iter first, Iter const& last,
		std::size_t width, boost::filesystem::path const& filepath, scalar ratio) {
	auto image = fipImage{FIT_BITMAP,
		static_cast<unsigned int>(width), static_cast<unsigned int>(width), 32};
	// coordinate system of freeimage is upside down.
	for(auto y = static_cast<int>(width)-1; y >= 0; --y) {
		BYTE* row = image.getScanLine(y);
		for(auto x = 0u; x < width; ++x) {
			assert(first != last);
			*(row+x*4) = *(row+x*4+1) = *(row+x*4+2) = ratio*(*first);
			++first;
		}
	}
	image.save(filepath.string().c_str());
	assert(first == last);
}
decltype(auto) save_1ch_image_vector_as_monochro_image(
		neu::cpu_vector const& image_vector,
		std::size_t width, boost::filesystem::path const& filepath, scalar ratio) {
	neu::save_1ch_image_vector_as_monochro_image(image_vector.begin(), image_vector.end(),
		width, filepath, ratio);
}
}// namespace neu

#endif //NEU_IMAGE_HPP
