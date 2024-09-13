#include "ImageUtils.hpp"
#include "GlobalParameters.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

Image::Image() : imageData(nullptr) {}

std::ifstream Image::openImageFile(const std::string &imagePath) {
    std::ifstream fbin(imagePath, std::ios::binary);
    if (!fbin) {
        throw std::runtime_error("Failed to open image file: " + imagePath);
    }
    return fbin;
}

std::unique_ptr<float[]> Image::readImageData(std::ifstream &fbin, int &height, int &width) {
    // Leer las dimensiones de la imagen
    readImageDimensions(fbin, height, width);

    // Calcular el tamaño del buffer y leer los datos de la imagen
    size_t size = height * width;
    std::unique_ptr<float[]> data(new float[size]);
    fbin.read(reinterpret_cast<char *>(data.get()), size * sizeof(float));
    return data;
}

void Image::readImageDimensions(std::ifstream &fbin, int &height, int &width) {
    fbin.read(reinterpret_cast<char *>(&height), sizeof(height));
    fbin.read(reinterpret_cast<char *>(&width), sizeof(width));
}

std::string Image::getExampleImagePath(int quality) {
    return "image" + convertImageTypeToString(quality) + ".bin";
}

std::unique_ptr<float[]> Image::loadImageData(int quality, int &height, int &width) {
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe").parent_path();
    std::string path = exe_path.string() + "/media/" + getExampleImagePath(quality);

    std::cout << "\n---------------------------------------------------------------------------------------\n";
    std::cout << " OPEN IMAGE";
    std::cout << "\n---------------------------------------------------------------------------------------\n";
    std::size_t found = path.find_last_of("/\\");
    std::string fileName = path.substr(found + 1);
    std::cout << " Open image file (bin): " << fileName << std::endl;
    std::ifstream fbin = openImageFile(path);

    imageData = readImageData(fbin, height, width);
    std::cout << " Height = " << height << "; Width = " << width
              << (VERBOSE_ENABLED ? "; Data size = " + std::to_string(static_cast<double>(height * width * sizeof(float)) / (1024.0 * 1024.0)) + " MB" : "")
              << std::endl;

    // Crear un nuevo std::unique_ptr con el tamaño adecuado
    std::unique_ptr<float[]> copy(new float[height * width]);

    // Copiar los datos de imageData al nuevo buffer
    std::copy(imageData.get(), imageData.get() + height * width, copy.get());

    return copy;
}

const std::unique_ptr<float[]> &Image::getImageData() const {
    return imageData;
}

const std::string Image::convertImageTypeToString(int type) const {
    switch (type) {
    case 1:
        return "1080p";
    case 2:
        return "1440p";
    case 3:
        return "2160p";
    case 4:
        return "2880p";
    case 5:
        return "4320p";
    default:
        return "640p";
    }
}
