#pragma once

#include <cstddef>
#include <vector>

namespace tinysplat {

/// Row-major image buffer: index (y, x, c) => ((y * width + x) * channels + c).
class Image {
 public:
  Image() = default;
  Image(int height, int width, int channels, float fill = 0.0f);

  int height() const { return height_; }
  int width() const { return width_; }
  int channels() const { return channels_; }
  std::size_t size() const { return data_.size(); }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }

  float& at(int y, int x, int c);
  float at(int y, int x, int c) const;

  void fill(float value);
  void resize(int height, int width, int channels, float fill = 0.0f);

 private:
  int height_ = 0;
  int width_ = 0;
  int channels_ = 0;
  std::vector<float> data_;
};

}  // namespace tinysplat
