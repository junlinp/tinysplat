#include "tinysplat/image.h"

#include <stdexcept>

namespace tinysplat {

Image::Image(int height, int width, int channels, float fill)
    : height_(height), width_(width), channels_(channels) {
  if (height < 0 || width < 0 || channels < 0) {
    throw std::invalid_argument("Image dimensions must be non-negative");
  }
  data_.assign(static_cast<std::size_t>(height) * width * channels, fill);
}

float& Image::at(int y, int x, int c) {
  return data_[static_cast<std::size_t>(((y * width_ + x) * channels_) + c)];
}

float Image::at(int y, int x, int c) const {
  return data_[static_cast<std::size_t>(((y * width_ + x) * channels_) + c)];
}

void Image::fill(float value) {
  std::fill(data_.begin(), data_.end(), value);
}

void Image::resize(int height, int width, int channels, float fill) {
  height_ = height;
  width_ = width;
  channels_ = channels;
  data_.assign(static_cast<std::size_t>(height) * width * channels, fill);
}

}  // namespace tinysplat
