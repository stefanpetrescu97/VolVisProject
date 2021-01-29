#include "renderer.h"
#include <algorithm>
#include <algorithm> // std::fill
#include <cmath>
#include <functional>
#include <glm/common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tuple>

//new library added for normalization
#include <glm/glm.hpp>

//#include <glm>

namespace render {

// The renderer is passed a pointer to the volume, gradinet volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);

    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multithreading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
        tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!instersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            };
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getVoxelInterpolate(samplePos);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getVoxelInterpolate(samplePos);
        maxVal = std::max(val, maxVal);
    }

    glm::vec3 test1 = glm::vec3(maxVal);
    glm::vec3 test2 = glm::vec3(maxVal) / m_pVolume->maximum();

    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

// ======= TODO: IMPLEMENT ========
// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function (to be implemented) to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{
    static constexpr glm::vec3 isoColor {0.8f, 0.8f, 0.2f};
    
    //current selected iso value; this can change depending on user input
    float isoValCurrent = m_config.isoValue;

    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    //precise value computed using bisection accuracy
    glm::vec3 preciseSamplePos;

    if(!m_config.volumeShading){
        //if volume shading is OFF
        //sample along the ray until an isosurface
        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
            const float val = m_pVolume->getVoxelInterpolate(samplePos);
            if(val > isoValCurrent){
                return glm::vec4(isoColor, 1.0f);
            }
        }
        //if no surface found for which the interpolated voxel value is greater than the isovalue => the pixel is not on the surface
        //return a pixel of 0 opacity
        return glm::vec4(0.0f);
    }else{
        //if volume shading is ON
        //sample along the ray until an isosurface
        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
            const float val = m_pVolume->getVoxelInterpolate(samplePos);
            if(val > isoValCurrent){
                //if found isosurface then => search for precise sample value using bisection
                float preciseValue = bisectionAccuracy(ray, t-sampleStep, t, isoValCurrent);
                preciseSamplePos = ray.origin + preciseValue * ray.direction;

                //get the gradient at the precise position
                volume::GradientVoxel gradient = m_pGradientVolume->getGradientVoxel(preciseSamplePos);

                //previous implementation without bisection => we simlpy get the gradient by using samplePos instead of preciseSamplePos
                //volume::GradientVoxel gradient = m_pGradientVolume->getGradientVoxel(samplePos);

                //return the shaded color
                return glm::vec4(computePhongShading(isoColor, gradient, m_pCamera->forward(), ray.direction), 1.0f);
            }
        }
        //if no surface found for which the interpolated voxel value is greater than the isovalue => the pixel is not on the surface
        //return a pixel of 0 opacity
        return glm::vec4(0.0f);
    }
}

// ======= TODO: IMPLEMENT ========
// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{   
    int numberOfIterations = 10;
    float thresholdDifference = 0.01;

    //step value exactly at the middle of [t0, t1] interval
    float middleOfInterval;

    //the value which will enable us to tell if the voxel value at step tMiddleInterval is > or < than isoValue
    float val;

    for(int i = 0; i < numberOfIterations; i++){
        //assign to middleOfInterval middle of interval [t0, t1]
        middleOfInterval = (t0 + t1)/2;

        //compute the voxel sample position at step tMiddleInterval
        glm::vec3 tMiddleSamplePos = ray.origin + middleOfInterval * ray.direction;

        //interpolate voxel value at tMiddleSamplePos sample position
        val = m_pVolume->getVoxelInterpolate(tMiddleSamplePos);

        //check end condition / change interval ends [t0, t1] for further iterations
        if(abs(val - isoValue) < thresholdDifference){
            break;
            //return middleOfInterval;
        }else if(val > isoValue){
            //t0 = t0;
            t1 = middleOfInterval;
        }else{
            t0 = middleOfInterval;
            //t1 = t1;
        }
    }

    //return best found step
    return middleOfInterval;
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.
glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{
    //refactored traceRayComposite() function in order to work with shading as well
    
    //initialize the sample position to the beginning of the volume
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

    //compute the size of the increment for each step
    const glm::vec3 increment = sampleStep * ray.direction;

    //initialize the accumulated color and opacity
    glm::vec4 comp_col = glm::vec4(0.0f);

    if(!m_config.volumeShading){
        //if volume shading is OFF
        //for every sample, accumulate the opacity and color
        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
            const float val = m_pVolume->getVoxelInterpolate(samplePos);
            glm::vec4 color = getTFValue(val);
            comp_col.r = comp_col.r + (1.0 - comp_col.a) * color.a * color.r;
            comp_col.g = comp_col.g + (1.0 - comp_col.a) * color.a * color.g;
            comp_col.b = comp_col.b + (1.0 - comp_col.a) * color.a * color.b;
            comp_col.a = comp_col.a + (1.0 - comp_col.a) * color.a;
            if (1-comp_col.a <= 0.01){
                return comp_col;
            }
        }
    
        return comp_col;

    }else{
        //if volume shading is ON
        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

            //get the current val for the voxel
            const float val = m_pVolume->getVoxelInterpolate(samplePos);

            //use the TF to get the color
            glm::vec4 color = getTFValue(val);

            //get the gradient at the sample position
            volume::GradientVoxel gradient = m_pGradientVolume->getGradientVoxel(samplePos);

            //check if its magnitude is smaller than a certain threshold (so for example if the gradient.magnitude is smaller than 0.0001 I consider it 0)
            if(gradient.magnitude < 0.0001){
                gradient.magnitude = 0;
            }

            //construct color vec3 in order to pass as parameter (as the current color vector is v4)
            glm::vec3 paramColorForShading(color.r, color.g, color.b);

            //get the shaded color by calling the computePhongShading (as the first two params I pass the previously computed gradient and the previously computed color)
            glm::vec3 shadedColor = computePhongShading(paramColorForShading, gradient, m_pCamera->forward(), ray.direction);
            
            //construct the shaded color, using the returned shaded RGB colors and also the previous color.a
            glm::vec4 finalShadedColor(shadedColor, color.a);

            color = finalShadedColor;

            //composite using the color returned by the computePhongShading function
            comp_col.r = comp_col.r + (1.0 - comp_col.a) * color.a * color.r;
            comp_col.g = comp_col.g + (1.0 - comp_col.a) * color.a * color.g;
            comp_col.b = comp_col.b + (1.0 - comp_col.a) * color.a * color.b;
            comp_col.a = comp_col.a + (1.0 - comp_col.a) * color.a;

            if (1-comp_col.a <= 0.01){
                return comp_col;
            }
        }
        return comp_col;
    }
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.
glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    //initialize the sample position to the beginning of the volume
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

    //compute the size of the increment for each step
    const glm::vec3 increment = sampleStep * ray.direction;

    //initialize the accumulated color and opacity
    glm::vec4 comp_op = glm::vec4(0.0f);

    //for every sample, accumulate the opacity
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getVoxelInterpolate(samplePos);
        const float magnitude = m_pGradientVolume->getGradientVoxel(samplePos).magnitude;
        float opacity = getTF2DOpacity(val, magnitude) * m_config.TF2DColor.w;

        comp_op.a = comp_op.a + (1.0f - comp_op.a) * opacity;
        comp_op.r = comp_op.r + (1-comp_op.a) * opacity * m_config.TF2DColor.r;
        comp_op.g = comp_op.g + (1-comp_op.a) * opacity * m_config.TF2DColor.g;
        comp_op.b = comp_op.b + (1-comp_op.a) * opacity * m_config.TF2DColor.b;

        if (1-comp_op.a <= 0.01){
            return comp_op;
        }
    }

    return comp_op;
}

// ======= TODO: IMPLEMENT ========
// Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector.
// You can find out more about the Phong shading model at:
// https://en.wikipedia.org/wiki/Phong_reflection_model
//
// Use the given color for the ambient/specular/diffuse (you are allowed to scale these constants by a scalar value).
// You are free to choose any specular power that you'd like.
glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V)
{
    //if gradient magnitude is null return a transparent voxel
    if(gradient.magnitude == 0){
        return glm::vec3(0);
    }

    //reflectiveness constants
    float ka = 0.1f;//ambient
    float kd = 0.7f;//diffuse
    float ks = 0.2f;//specular
    float a = 100;

    //set the colors; compute the 3 bands separately
    float ir = (float) color.r;
    float ig = (float) color.g;
    float ib = (float) color.b;
    

    // compute light vector
    glm::vec3 toLight((float) -L[0], (float) -L[1], (float) -L[2]);
    glm::vec3 toLightN = glm::normalize(toLight);

    // compute view vector
    glm::vec3 toView((float) -V[0], (float) -V[1], (float) -V[2]);
    glm::vec3 toViewN = glm::normalize(toView);

    //compute normal vector
    glm::vec3 normal((float) -gradient.dir.x, (float) -gradient.dir.y, (float) -gradient.dir.z);
    glm::vec3 normalN = glm::normalize(normal);

    //compute light reflection vector
    float dotp = glm::dot(toLightN, normalN);
    glm::vec3 scaled = normalN * (2*dotp);

    // rN is the the direction taken by a perfect reflection of the light source on the surface
    glm::vec3 rN = scaled - toLightN;

    //store ambient color
    float r_ambient = ka * ir;
    float g_ambient = ka * ig;
    float b_ambient = ka * ib;

    //check if normal is in correct direction, if light is orthogonal(or larger angle) to the surface only use ambient lighting
    if(glm::atan(glm::acos(glm::dot(toLight, normal))) >= 90){
        return glm::vec3(r_ambient,g_ambient,b_ambient);
    }

    //store diffuse color
    float r_diffuse = kd * glm::dot(toLightN, normalN) * ir;
    float g_diffuse = kd * glm::dot(toLightN, normalN) * ig;
    float b_diffuse = kd * glm::dot(toLightN, normalN) * ib;

    //final step in computing the specular light reflection
    float specPow =  (float) pow(glm::dot(rN, toViewN), a);

    //store specular color
    float r_specular = ks * specPow * ir;
    float g_specular = ks * specPow * ig;
    float b_specular = ks * specPow * ib;

    //store the final color
    float newColorR = r_ambient + r_diffuse + r_specular;
    float newColorG = g_ambient + g_diffuse + g_specular;
    float newColorB = b_ambient + b_diffuse + b_specular;

    //clamp the color values if >1 || <0
    if(newColorR > 1){
        newColorR = 1;
    }else if(newColorR < 0){
        newColorR = 0;
    }
    if(newColorG > 1){
        newColorG = 1;
    }else if(newColorG < 0){
        newColorG = 0;
    }
    if(newColorB > 1){
        newColorB = 1;
    }else if(newColorB < 0){
        newColorB = 0;
    }

    //keep transparency of color passed as argument
    glm::vec3 resultColor = glm::vec3(newColorR,newColorG,newColorB);

    return resultColor;
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D tranfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size()-1);
    return m_config.tfColorMap[i];
}

// ======= TODO: IMPLEMENT ========
// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f
//
// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{
    /*float radius = m_config.TF2DRadius;
    float center = m_config.TF2DIntensity;

    float radiusatgradient = radius * (gradientMagnitude / 256.0f);
    // std::cout << gradientMagnitude << std::endl;
    if (intensity < center - radiusatgradient || intensity > center + radiusatgradient) {
        return 0.0f;
    }

    intensity = abs(intensity - center);
    float res = 1.0f - (intensity/radiusatgradient);

    if (res <= 0.0f) {
        return 0.0f;
    }
    return res;*/
    
    double opacity = 0.0;

    // retrieve widget data
    const float radius = m_config.TF2DRadius;
    const float maxmag = m_pGradientVolume->maxMagnitude();

    // compute the widget angle between apex and side of triangle
    const double wid_angle = glm::atan(radius/maxmag);
    // compute point angle
    double point_angle = glm::atan(glm::abs(intensity-m_config.TF2DIntensity)/gradientMagnitude);
    
    // if the point is not inside the angle or if its gradient is negative, just return 0
    if(point_angle < wid_angle && gradientMagnitude > 0){
        opacity = 1-(point_angle/wid_angle);
    }
    
    return opacity;
}

// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::instersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}
}