#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <fstream>
#include <memory>
#include <string>
#include <vector>

class Image {
  public:
    Image();
    ~Image() = default;

    std::ifstream openImageFile(const std::string &imagePath);
    std::unique_ptr<float[]> readImageData(std::ifstream &fbin, int &height, int &width);
    void readImageDimensions(std::ifstream &fbin, int &height, int &width);
    std::string getExampleImagePath(int quality);
    std::unique_ptr<float[]> loadImageData(int quality, int &height, int &width);
    const std::unique_ptr<float[]> &getImageData() const;
    const std::string convertImageTypeToString(int type) const;

  private:
    std::unique_ptr<float[]> imageData;
};

#endif // IMAGE_UTILS_HPP