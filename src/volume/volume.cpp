#include "volume.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype> // isspace
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <gsl/span>
#include <iostream>
#include <string>

struct Header {
    glm::ivec3 dim;
    size_t elementSize;
};
static Header readHeader(std::ifstream& ifs);
static float computeMinimum(gsl::span<const uint16_t> data);
static float computeMaximum(gsl::span<const uint16_t> data);
static std::vector<int> computeHistogram(gsl::span<const uint16_t> data);

namespace volume {

Volume::Volume(const std::filesystem::path& file)
    : m_fileName(file.string())
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    loadFile(file);
    auto end = clock::now();
    std::cout << "Time to load: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;

    if (m_data.size() > 0) {
        m_minimum = computeMinimum(m_data);
        m_maximum = computeMaximum(m_data);
        m_histogram = computeHistogram(m_data);
    }
}

Volume::Volume(std::vector<uint16_t> data, const glm::ivec3& dim)
    : m_fileName()
    , m_elementSize(2)
    , m_dim(dim)
    , m_data(std::move(data))
    , m_minimum(computeMinimum(m_data))
    , m_maximum(computeMaximum(m_data))
    , m_histogram(computeHistogram(m_data))
{
}

float Volume::minimum() const
{
    return m_minimum;
}

float Volume::maximum() const
{
    return m_maximum;
}

std::vector<int> Volume::histogram() const
{
    return m_histogram;
}

glm::ivec3 Volume::dims() const
{
    return m_dim;
}

std::string_view Volume::fileName() const
{
    return m_fileName;
}

float Volume::getVoxel(int x, int y, int z) const
{
    const size_t i = size_t(x + m_dim.x * (y + m_dim.y * z));
    return static_cast<float>(m_data[i]);
}

// This function returns a value based on the current interpolation mode
float Volume::getVoxelInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getVoxelNN(coord);
    }
    case InterpolationMode::Linear: {
        return getVoxelLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        return getVoxelTriCubicInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    }
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
float Volume::getVoxelNN(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord + 0.5f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 0.5f, glm::vec3(m_dim))))
        return 0.0f;

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getVoxel(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// This function returns the trilinear interpolated value of the position given by position coord.
float Volume::getVoxelLinearInterpolate(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim - 1))))
        return 0.0f;

    const int x = static_cast<int>(coord.x);
    const int y = static_cast<int>(coord.y);
    const int z = static_cast<int>(coord.z);

    const float fac_x = coord.x - float(x);
    const float fac_y = coord.y - float(y);
    const float fac_z = coord.z - float(z);

    const float t0 = linearInterpolate(getVoxel(x, y, z), getVoxel(x + 1, y, z), fac_x);
    const float t1 = linearInterpolate(getVoxel(x, y + 1, z), getVoxel(x + 1, y + 1, z), fac_x);
    const float t2 = linearInterpolate(getVoxel(x, y, z + 1), getVoxel(x + 1, y, z + 1), fac_x);
    const float t3 = linearInterpolate(getVoxel(x, y + 1, z + 1), getVoxel(x + 1, y + 1, z + 1), fac_x);
    const float t4 = linearInterpolate(t0, t1, fac_y);
    const float t5 = linearInterpolate(t2, t3, fac_y);
    const float t6 = linearInterpolate(t4, t5, fac_z);
    return t6;
}

// This function linearly interpolates the value g0 and g1 given a factor
// The result is returned. It is used for the tri-linearly interpolation the values
float Volume::linearInterpolate(float g0, float g1, float factor)
{
    return (1 - factor) * g0 + factor * g1;
}

// ======= TODO : IMPLEMENT ========
// This function represents the h(x) function, which returns the weight of the cubic interpolation kernel for a given position x
float Volume::weight(float x)
{
    float l = 0.0;
    const float a = -0.75;

    x = abs(x);
    if (x>=0 && x<1){
        l = (a+2)* pow(x,3) - (a+3)*pow(x,2) +1;
    }
    else if (x>=1 && x<2)
    {
        l = a*pow(x,3) - 5*a*pow(x,2) + 8*a*x - 4*a;
    }
    return l;
}

// ======= TODO : IMPLEMENT ========
// This functions returns the results of a cubic interpolation using 4 values and a factor
float Volume::cubicInterpolate(float g0, float g1, float g2, float g3, float factor)
{
    //g1 is the value of the point at distance factor from x
    float w0 = weight(1 + factor);
    float w1 = weight(factor);
    float w2 = weight(1 - factor);
    float w3 = weight(2 - factor);

    float res = w0*g0 + w1*g1 + w2*g2 + w3*g3;

    return res;
}

// ======= TODO : IMPLEMENT ========
// This function returns the value of a bicubic interpolation
float Volume::bicubicInterpolateXY(const glm::vec2& xyCoord, int z) const
{
    const int x = static_cast<int>(xyCoord.x);
    const int y = static_cast<int>(xyCoord.y);

    const float fac_x = xyCoord.x - float(x);
    const float fac_y = xyCoord.y - float(y);

    // First we interpolate along the x-axis, for the 4 surrounding values of y
    const float v0 = cubicInterpolate(getVoxel(x-1,y-1,z), getVoxel(x,y-1,z), getVoxel(x+1,y-1,z), getVoxel(x+2,y-1,z), fac_x);
    const float v1 = cubicInterpolate(getVoxel(x-1,y,z), getVoxel(x,y,z), getVoxel(x+1,y,z), getVoxel(x+2,y,z), fac_x);
    const float v2 = cubicInterpolate(getVoxel(x-1,y+1,z), getVoxel(x,y+1,z), getVoxel(x+1,y+1,z), getVoxel(x+2,y+1,z), fac_x);
    const float v3 = cubicInterpolate(getVoxel(x-1,y+2,z), getVoxel(x,y+2,z), getVoxel(x+1,y+2,z), getVoxel(x+2,y+2,z), fac_x);

    // Then we interpolate the resulting 4 values along the y-axis
    const float res = cubicInterpolate(v0, v1, v2, v3, fac_y);

    return res;
}

// ======= TODO : IMPLEMENT ========
// This function computes the tricubic interpolation at coord
float Volume::getVoxelTriCubicInterpolate(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(1))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim - 3))))
        return 0.0f;

    glm::vec2 xy = glm::vec2(coord);
    const int z = static_cast<int>(coord.z);

    const float fac_z = coord.z - float(z);

    const float z0 = bicubicInterpolateXY(xy, z-1);
    const float z1 = bicubicInterpolateXY(xy, z);
    const float z2 = bicubicInterpolateXY(xy, z+1);
    const float z3 = bicubicInterpolateXY(xy, z+2);

    const float res = cubicInterpolate(z0, z1, z2, z3, fac_z);

    if (res < 0){ return 0.0f;}
    else if(res > 255){return 255.0f;}

    return res;
}

// Load an fld volume data file
// First read and parse the header, then the volume data can be directly converted from bytes to uint16_ts
void Volume::loadFile(const std::filesystem::path& file)
{
    assert(std::filesystem::exists(file));
    std::ifstream ifs(file, std::ios::binary);
    assert(ifs.is_open());

    const auto header = readHeader(ifs);
    m_dim = header.dim;
    m_elementSize = header.elementSize;

    const size_t voxelCount = static_cast<size_t>(header.dim.x * header.dim.y * header.dim.z);
    const size_t byteCount = voxelCount * header.elementSize;
    std::vector<char> buffer(byteCount);
    // Data section is separated from header by two /f characters.
    ifs.seekg(2, std::ios::cur);
    ifs.read(buffer.data(), std::streamsize(byteCount));

    m_data.resize(voxelCount);
    if (header.elementSize == 1) { // Bytes.
        for (size_t i = 0; i < byteCount; i++) {
            m_data[i] = static_cast<uint16_t>(buffer[i] & 0xFF);
        }
    } else if (header.elementSize == 2) { // uint16_ts.
        for (size_t i = 0; i < byteCount; i += 2) {
            m_data[i / 2] = static_cast<uint16_t>((buffer[i] & 0xFF) + (buffer[i + 1] & 0xFF) * 256);
        }
    }
}
}

static Header readHeader(std::ifstream& ifs)
{
    Header out {};

    // Read input until the data section starts.
    std::string line;
    while (ifs.peek() != '\f' && !ifs.eof()) {
        std::getline(ifs, line);
        // Remove comments.
        line = line.substr(0, line.find('#'));
        // Remove any spaces from the string.
        // https://stackoverflow.com/questions/83439/remove-spaces-from-stdstring-in-c
        line.erase(std::remove_if(std::begin(line), std::end(line), ::isspace), std::end(line));
        if (line.empty())
            continue;

        const auto separator = line.find('=');
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);

        if (key == "ndim") {
            if (std::stoi(value) != 3) {
                std::cout << "Only 3D files supported\n";
            }
        } else if (key == "dim1") {
            out.dim.x = std::stoi(value);
        } else if (key == "dim2") {
            out.dim.y = std::stoi(value);
        } else if (key == "dim3") {
            out.dim.z = std::stoi(value);
        } else if (key == "nspace") {
        } else if (key == "veclen") {
            if (std::stoi(value) != 1)
                std::cerr << "Only scalar m_data are supported" << std::endl;
        } else if (key == "data") {
            if (value == "byte") {
                out.elementSize = 1;
            } else if (value == "short") {
                out.elementSize = 2;
            } else {
                std::cerr << "Data type " << value << " not recognized" << std::endl;
            }
        } else if (key == "field") {
            if (value != "uniform")
                std::cerr << "Only uniform m_data are supported" << std::endl;
        } else if (key == "#") {
            // Comment.
        } else {
            std::cerr << "Invalid AVS keyword " << key << " in file" << std::endl;
        }
    }
    return out;
}

static float computeMinimum(gsl::span<const uint16_t> data)
{
    return float(*std::min_element(std::begin(data), std::end(data)));
}

static float computeMaximum(gsl::span<const uint16_t> data)
{
    return float(*std::max_element(std::begin(data), std::end(data)));
}

static std::vector<int> computeHistogram(gsl::span<const uint16_t> data)
{
    std::vector<int> histogram(size_t(*std::max_element(std::begin(data), std::end(data)) + 1), 0);
    for (const auto v : data)
        histogram[v]++;
    return histogram;
}
