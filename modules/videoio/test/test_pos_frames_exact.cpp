// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

// big_buck_bunny.mp4: mpeg4, 24fps, 125 frames, key frame every 12 frames (0,12,...,120).
static string posFramesExactTestVideoPath()
{
    return findDataFile("video/big_buck_bunny.mp4");
}

// FFmpeg's own seek-then-decode-forward already lands exactly; confirm the generic layer agrees.
TEST(videoio_pos_frames_exact, ffmpeg_non_raw_seek_is_always_exact)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_FFMPEG);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 1, 5, 12, 15, 20, 24, 62, 100, 124})
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// RAW mode has no codec to decode forward with, so a seek is exact only when the target is itself a key frame.
TEST(videoio_pos_frames_exact, ffmpeg_raw_mode_seek_is_exact_only_on_key_frames)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_FFMPEG, {CAP_PROP_FORMAT, -1});
    ASSERT_TRUE(cap.isOpened());

    struct { int target; int landed; bool exact; } cases[] = {
        {0, 0, true}, {12, 12, true}, {24, 24, true},    // requested frame is itself a key frame
        {5, 0, false}, {15, 12, false}, {20, 12, false}, // between key frames 0 and 12
        {62, 60, false}, {100, 96, false}, {124, 120, false},
    };
    for (const auto& c : cases)
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, c.target)) << "target " << c.target;
        // Checked before read(): a raw-mode grab right after a seek can leave the reported position unchanged.
        EXPECT_EQ(c.landed, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << c.target;
        EXPECT_EQ(c.exact ? 1 : 0, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << c.target;
        Mat raw;
        ASSERT_TRUE(cap.read(raw)) << "target " << c.target;
    }
}

// Sweeps every frame in the file, in both modes, to confirm the backend's key-frame seek never overshoots (grabFrame() can't correct an overshoot).
TEST(videoio_pos_frames_exact, ffmpeg_seek_never_lands_past_the_target)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    for (bool raw : {false, true})
    {
        VideoCapture cap = raw
            ? VideoCapture(posFramesExactTestVideoPath(), CAP_FFMPEG, {CAP_PROP_FORMAT, -1})
            : VideoCapture(posFramesExactTestVideoPath(), CAP_FFMPEG);
        ASSERT_TRUE(cap.isOpened()) << "raw=" << raw;

        for (int target = 0; target < 125; target++)
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "raw=" << raw << " target=" << target;
            double landed = cap.get(CAP_PROP_POS_FRAMES);
            ASSERT_NE(static_cast<double>(CAP_PROP_UNKNOWN), landed) << "raw=" << raw << " target=" << target;
            EXPECT_LE(cvRound(landed), target) << "raw=" << raw << " target=" << target;
            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << "raw=" << raw << " target=" << target;
        }
    }
}

// Intra-only codec: every frame is its own key frame, so this is CAP_IMAGES-like -- always exact.
TEST(videoio_pos_frames_exact, opencv_mjpeg_seek_is_always_exact)
{
    if (!videoio_registry::hasBackend(CAP_OPENCV_MJPEG))
        throw SkipTestException("CAP_OPENCV_MJPEG backend was not found");

    VideoCapture cap(findDataFile("video/big_buck_bunny.mjpg.avi"), CAP_OPENCV_MJPEG);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 1, 5, 12, 24, 62, 100, 124})
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// A synthetic pipeline avoids qtdemux specifics, but position querying here is still consistently unverifiable in practice.
static std::string posFramesExactGstreamerPipeline(int srcFrameCount, double srcFps)
{
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << srcFrameCount
             << " ! video/x-raw,framerate=" << cvRound(srcFps) << "/1 ! appsink";
    return pipeline.str();
}

// A seek this backend accepts but can't verify must report -1, never a false 1; a verified one must match where it landed.
TEST(videoio_pos_frames_exact, gstreamer_synthetic_pipeline_seek_stays_honest_either_way)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(posFramesExactGstreamerPipeline(30, 30.0), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());

    Mat frame;
    ASSERT_TRUE(cap.read(frame)); // prime: matches the preroll state a real caller would be in

    for (int target : {0, 5, 10, 20, 29})
    {
        bool ok = cap.set(CAP_PROP_POS_FRAMES, target);
        double exact = cap.get(CAP_PROP_POS_FRAMES_IS_EXACT);
        if (!ok)
        {
            EXPECT_NE(1, cvRound(exact)) << "target " << target;
            continue;
        }
        if (cvRound(exact) == 1)
        {
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        }
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// #10324: seeking on a real container is unreliable in ways the generic layer can't fix; this only checks the contract stays honest either way.
TEST(videoio_pos_frames_exact, gstreamer_real_file_seek_stays_honest_either_way)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 5, 15, 62, 100})
    {
        bool ok = cap.set(CAP_PROP_POS_FRAMES, target);
        double exact = cap.get(CAP_PROP_POS_FRAMES_IS_EXACT);
        // Checked before read(): GStreamer's own position reporting can become unavailable again after a subsequent read().
        if (!ok)
        {
            // A failed seek must never claim to be exact.
            EXPECT_NE(1, cvRound(exact)) << "target " << target;
        }
        else if (cvRound(exact) == 1)
        {
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        }
        if (ok)
        {
            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << "target " << target;
        }
    }
}

// Each encoded frame is a flat, distinct color so a decoded frame's content can be matched back
// to its source index -- this is what lets gstreamer_read_after_seek_does_not_skip_the_landed_frame
// below tell "read() returned the frame the seek landed on" apart from "read() returned the next one".
static Scalar gstreamerGopFixtureFrameColor(int i)
{
    return Scalar(i * 5 % 256, (i * 7 + 30) % 256, (i * 11 + 60) % 256);
}

// A synthetic file with a real GOP structure (unlike the live pipelines above) so a non-key-frame seek can exercise decode-forward correction for real, not just stay honest about not verifying.
static std::string generateGstreamerGopFixture()
{
    std::string path = cv::tempfile(".mp4");
    std::string pipeline = "appsrc ! videoconvert ! x264enc key-int-max=12 bframes=0 ! qtmux ! filesink location=" + path;
    VideoWriter writer(pipeline, CAP_GSTREAMER, 0, 24.0, Size(64, 48), true);
    if (!writer.isOpened())
        return std::string();
    for (int i = 0; i < 30; i++)
        writer.write(Mat(48, 64, CV_8UC3, gstreamerGopFixtureFrameColor(i)));
    return path;
}

TEST(videoio_pos_frames_exact, gstreamer_decode_forward_corrects_at_least_one_landing)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGstreamerGopFixture();
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    int exactNonKeyFrameCount = 0;
    for (int target : {1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23})
    {
        // A fresh capture per target sidesteps a separate, unrelated issue where many consecutive seeks on one GStreamer capture can eventually stall.
        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened()) << "target " << target;
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        int exact = cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT));
        int landed = cvRound(cap.get(CAP_PROP_POS_FRAMES));
        if (exact == 1)
        {
            EXPECT_EQ(target, landed) << "target " << target;
            exactNonKeyFrameCount++;
        }
        else
        {
            EXPECT_NE(target, landed) << "target " << target << ": landed exactly but wasn't reported as exact";
        }
        Mat frame;
        EXPECT_TRUE(cap.read(frame)) << "target " << target;
    }

    EXPECT_GT(exactNonKeyFrameCount, 0)
        << "decode-forward correction never landed exactly on any non-key-frame target across the sweep";

    remove(path.c_str());
}

// Regression test for a bug found while verifying this feature manually (not caught by either
// GitHub review round, and not caught by the test above): correctPosFramesLanding()'s own
// decode-forward loop grabs frames internally to walk up to the target, and the grab that lands
// exactly on the target must be left for the caller's own read() to consume -- not silently
// discarded. Before the fix, that landing grab's sample was thrown away and the *following*
// grabFrame() call (i.e. the grab() half of the caller's read()) pulled a fresh sample one frame
// past the target -- so read() silently returned the wrong frame's pixels while
// CAP_PROP_POS_FRAMES_IS_EXACT still confidently reported 1. Checking only the reported position
// (as every other test in this file does) can't see this bug: the position was reported correctly
// the whole time, only the decoded pixel content was wrong. This test decodes the frame and checks
// its actual color against the target index, not the index after it.
TEST(videoio_pos_frames_exact, gstreamer_read_after_seek_does_not_skip_the_landed_frame)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGstreamerGopFixture();
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    // Non-monotonic on purpose: an earlier seek priming the pipeline, then non-key-frame targets
    // that require decode-forward, mirroring the real seek pattern the bug was originally found
    // under (repeated seeks on one capture, not just a single isolated seek).
    int targets[] = {5, 2, 8, 4, 11, 19, 23};

    VideoCapture cap(path, CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());

    for (int target : targets)
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        if (cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)) != 1)
            continue; // only checking cases the backend itself claims are exact

        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;

        Scalar meanColor = mean(frame);
        Scalar expected = gstreamerGopFixtureFrameColor(target);
        Scalar nextFrameColor = gstreamerGopFixtureFrameColor(target + 1);

        // The regression: a skipped frame lands much closer to target+1's color than target's.
        double distToTarget = cv::norm(meanColor - expected, NORM_L1);
        double distToNext = cv::norm(meanColor - nextFrameColor, NORM_L1);
        EXPECT_LT(distToTarget, distToNext)
            << "target " << target << ": read() returned frame " << (target + 1)
            << "'s content instead of the frame the seek landed on (exact=1 was reported)";
        EXPECT_LE(distToTarget, 15.0)
            << "target " << target << ": decoded color too far from the expected flat color "
            << "for this target frame (encoder noise budget)";
    }

    cap.release();
    remove(path.c_str());
}

}} // namespace
