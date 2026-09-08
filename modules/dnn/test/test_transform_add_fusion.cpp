// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#include "test_transform_add_fusion.inc.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

// Model: ConvTranspose2d(8->16, C0=8 so 16 % C0 == 0) -> Add(residual). Exercises
// graph_fusion_transform_add.cpp: TransformLayout should absorb the Add, leaving no
// separate NaryEltwise layer, and the fused output must match doing the deinterleave
// and the add as two separate steps.
TEST(Layer_TransformLayoutAddFusion, FusedMatchesUnfusedComputation)
{
    Net net = readNetFromONNX((const char*)g_deconvAddOnnx, g_deconvAddOnnx_len);
    ASSERT_FALSE(net.empty());

    std::vector<std::string> preTypes;
    net.getLayerTypes(preTypes);
    ASSERT_NE(std::find(preTypes.begin(), preTypes.end(), std::string("NaryEltwise")), preTypes.end())
        << "test model must start with a separate NaryEltwise layer for this test to mean anything";

    RNG rng(12345);
    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    rng.fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 16, 12, 12}, CV_32F);
    rng.fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    // The raw ConvTranspose2 output, fetched by name before the fused Add runs on
    // it -- net.forward(out, name) hands back a plain, already-converted tensor
    // (not the layer's raw internal block-layout buffer), so this is directly
    // comparable to the final result with no layout handling needed here.
    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");

    Mat fusedOut = net.forward();

    // The fusion should have run: no NaryEltwise left in the graph.
    std::vector<std::string> postTypes;
    net.getLayerTypes(postTypes);
    EXPECT_EQ(std::find(postTypes.begin(), postTypes.end(), std::string("NaryEltwise")), postTypes.end())
        << "NaryEltwise should have been absorbed into TransformLayout by the fusion pass";

    // Unfused reference: the same two values added as a separate step, instead of
    // in one fused TransformLayout(+residual) pass.
    Mat unfusedOut;
    cv::add(deconvOut, residual, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused TransformLayout+Add", 1e-5, 1e-5);
}

// Model: ConvTranspose2d(4->8), out_channels == C0 (so NK1 == N*K1 == 1) and a
// spatial output of 100 elements -- computeSpatChunks() always splits this
// regardless of thread count, so the new spatial decomposition in deconvBlock32f()
// is guaranteed to run, not just the pre-existing single-chunk path.
TEST(Layer_DeconvolutionSpatialChunking, NarrowOutputWideSpatialMatchesReference)
{
    Net net = readNetFromONNX((const char*)g_deconvNarrowOnnx, g_deconvNarrowOnnx_len);
    ASSERT_FALSE(net.empty());

    Mat input(std::vector<int>{1, 4, 5, 5}, CV_32F, (void*)g_deconvNarrowInput);
    Mat expected(std::vector<int>{1, 8, 10, 10}, CV_32F, (void*)g_deconvNarrowExpected);

    net.setInput(input);
    Mat out = net.forward();

    normAssert(expected, out, "deconv output with nSpatChunks > 1", 1e-4, 1e-4);
}

}} // namespace
