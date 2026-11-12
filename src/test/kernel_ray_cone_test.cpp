/* SPDX-License-Identifier: Apache-2.0 */
#include <gtest/gtest.h>
#include <limits>
#include "kernel/util/ray_cone.h"

CCL_NAMESPACE_BEGIN

TEST(KernelRayCone, FootprintResolutionAndIncidence)
{
  const float2 dx = make_float2(.01f, 0.0f);
  const float2 dy = make_float2(0.0f, .01f);
  float lod = 0;
  ASSERT_TRUE(ray_cone_texture_lod(dx, dy, 100, 100, 1, &lod));
  EXPECT_NEAR(lod, 0, 1e-5f);
  ASSERT_TRUE(ray_cone_texture_lod(dx, dy, 100, 100, .25f, &lod));
  EXPECT_NEAR(lod, 2, 1e-5f);
  ASSERT_TRUE(ray_cone_texture_lod(dx * 2, dy * 2, 100, 100, 1, &lod));
  EXPECT_NEAR(lod, 1, 1e-5f);
  ASSERT_TRUE(ray_cone_texture_lod(dx, dy, 200, 200, 1, &lod));
  EXPECT_NEAR(lod, 1, 1e-5f);
  ASSERT_TRUE(ray_cone_texture_lod(-dx, dy, 100, 100, -.25f, &lod));
  EXPECT_NEAR(lod, 2, 1e-5f);
}

TEST(KernelRayCone, SingularMappingsAndGrazingAreSafe)
{
  const float2 dx = make_float2(.01f, 0.0f);
  const float2 dy = make_float2(0.0f, .01f);
  float lod = 17;
  EXPECT_FALSE(ray_cone_texture_lod(dx, dx, 100, 100, 1, &lod));
  EXPECT_EQ(lod, 17); // Caller retains its legacy LOD.
  EXPECT_FALSE(ray_cone_texture_lod(dx, dy, 100, 100,
                                  std::numeric_limits<float>::quiet_NaN(), &lod));
  ASSERT_TRUE(ray_cone_texture_lod(dx, dy, 100, 100, 0, &lod));
  EXPECT_TRUE(std::isfinite(lod));
}

CCL_NAMESPACE_END
