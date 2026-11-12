#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0,binding=0,std430) readonly buffer Pixels { uvec2 pixels[]; };
layout(push_constant) uniform Parameters { uint size; } parameters;
void main() {
  uint size = parameters.size;
  uvec2 p = min(uvec2(vec2(uv.x, 1.0-uv.y) * size), uvec2(size-1));
  uvec2 bits = pixels[p.y * size + p.x];
  vec3 c = max(vec3(unpackHalf2x16(bits.x), unpackHalf2x16(bits.y).x), 0.0);
  c = c / (1.0+c);
  c = mix(1.055 * pow(c, vec3(1.0/2.4)) - 0.055, 12.92*c,
          lessThanEqual(c, vec3(0.0031308)));
  color = vec4(c, 1.0);
}
