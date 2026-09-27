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

// Sweeps every frame in the file, in both modes: landing past the target would make set() fail.
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

static std::vector<Mat> readAllFrames(const std::string& path, int api, int maxFrames)
{
    std::vector<Mat> frames;
    VideoCapture cap(path, api);
    Mat frame;
    while ((int)frames.size() < maxFrames && cap.read(frame))
        frames.push_back(frame.clone());
    return frames;
}

// A timestamp seek to the start of this MPEG-PS file lands on frame 12, so early targets need a verified rewind.
TEST(videoio_pos_frames_exact, ffmpeg_seek_near_start_of_mpeg_ps_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const std::string path = findDataFile("video/big_buck_bunny.mpg");
    std::vector<Mat> reference = readAllFrames(path, CAP_FFMPEG, 16);
    ASSERT_EQ(16u, reference.size());

    for (int target = 0; target < 16; target++)
    {
        VideoCapture cap(path, CAP_FFMPEG);
        ASSERT_TRUE(cap.isOpened());
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
        EXPECT_EQ(0, cvtest::norm(frame, reference[target], NORM_INF)) << "target " << target;
    }
}

// This AVI can't be rewound verifiably, so a seek must report unknown rather than a wrong exact landing.
TEST(videoio_pos_frames_exact, ffmpeg_unverifiable_seek_does_not_claim_exact)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const std::string path = findDataFile("video/VID00003-20100701-2204.avi");
    std::vector<Mat> reference = readAllFrames(path, CAP_FFMPEG, 16);
    ASSERT_EQ(16u, reference.size());

    for (int target = 0; target < 16; target++)
    {
        VideoCapture cap(path, CAP_FFMPEG);
        ASSERT_TRUE(cap.isOpened());
        cap.set(CAP_PROP_POS_FRAMES, target);
        const bool claimsExact = cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)) == 1;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
        if (claimsExact)
        {
            EXPECT_EQ(0, cvtest::norm(frame, reference[target], NORM_INF)) << "target " << target;
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

static std::string posFramesExactGstreamerPipeline(int srcFrameCount, double srcFps)
{
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << srcFrameCount
             << " ! video/x-raw,framerate=" << cvRound(srcFps) << "/1 ! appsink";
    return pipeline.str();
}

// videotestsrc has no duration: only a seek to frame 0 works, and a failed seek must not claim exact.
TEST(videoio_pos_frames_exact, gstreamer_pipeline_without_duration_only_seeks_to_start)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(posFramesExactGstreamerPipeline(30, 30.0), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());

    Mat frame;
    ASSERT_TRUE(cap.read(frame));

    for (int target : {0, 5, 0, 20})
    {
        if (target == 0)
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target));
            EXPECT_EQ(0, cvRound(cap.get(CAP_PROP_POS_FRAMES)));
            EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)));
        }
        else
        {
            EXPECT_FALSE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
            EXPECT_EQ(-1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        }
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

TEST(videoio_pos_frames_exact, gstreamer_real_file_seek_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 5, 15, 62, 100})
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

static const int gstreamerGopFixtureFrameCount = 30;

// Each encoded frame is a flat, distinct color.
static Scalar gstreamerGopFixtureFrameColor(int i)
{
    return Scalar(i * 5 % 256, (i * 7 + 30) % 256, (i * 11 + 60) % 256);
}

// Index of the closest frame in `reference`, a sequential decode by the same backend.
static int nearestFrameIndex(const Mat& frame, const std::vector<Mat>& reference)
{
    int best = -1;
    double bestDist = DBL_MAX;
    for (size_t i = 0; i < reference.size(); i++)
    {
        double dist = cv::norm(frame, reference[i], NORM_L1);
        if (dist < bestDist)
        {
            bestDist = dist;
            best = (int)i;
        }
    }
    return best;
}

// By default a file with a key frame every 12 frames, so most seek targets sit between key frames.
static std::string generateGstreamerGopFixture(const std::string& x264Options = "key-int-max=12 bframes=0")
{
    std::string path = cv::tempfile(".mp4");
    std::string pipeline = "appsrc ! videoconvert ! x264enc " + x264Options + " ! h264parse ! qtmux ! filesink location=" + path;
    VideoWriter writer(pipeline, CAP_GSTREAMER, 0, 24.0, Size(64, 48), true);
    if (!writer.isOpened())
        return std::string();
    for (int i = 0; i < gstreamerGopFixtureFrameCount; i++)
        writer.write(Mat(48, 64, CV_8UC3, gstreamerGopFixtureFrameColor(i)));
    return path;
}

// Seeks back and forth on one capture must report the requested frame and read back its content.
TEST(videoio_pos_frames_exact, gstreamer_seek_lands_on_the_requested_frame)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGstreamerGopFixture();
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    std::vector<Mat> reference;
    {
        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened());
        Mat frame;
        while (cap.read(frame))
            reference.push_back(frame.clone());
    }
    ASSERT_EQ((size_t)gstreamerGopFixtureFrameCount, reference.size());

    {
        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened());

        for (int target : {5, 2, 8, 4, 11, 19, 23, 1, 13, 0, 17, 22, 12})
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
            EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;

            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << "target " << target;
            EXPECT_EQ(target, nearestFrameIndex(frame, reference)) << "target " << target;
        }
    }

    remove(path.c_str());
}

// Every frame, in shuffled order on one capture, must be exact and read back correctly.
TEST(videoio_pos_frames_exact, gstreamer_seek_to_every_frame_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    for (const char* name : {"video/rotated_metadata.mp4", "video/big_buck_bunny.mp4"})
    {
        const std::string path = findDataFile(name);
        std::vector<Mat> reference = readAllFrames(path, CAP_GSTREAMER, 200);
        ASSERT_FALSE(reference.empty()) << name;

        std::vector<int> targets(reference.size());
        std::iota(targets.begin(), targets.end(), 0);
        RNG rng(42);
        randShuffle(targets, 1.0, &rng);

        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened()) << name;
        for (int target : targets)
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << name << " target " << target;
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << name << " target " << target;
            EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << name << " target " << target;
            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << name << " target " << target;
            EXPECT_EQ(target, nearestFrameIndex(frame, reference)) << name << " target " << target;
        }
    }
}

// Open GOP decoding can yield corrupted frames; a seek that claims exact must still deliver the right frame.
TEST(videoio_pos_frames_exact, gstreamer_open_gop_seek_does_not_claim_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGstreamerGopFixture("key-int-max=12 bframes=3 option-string=open-gop=1");
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    std::vector<Mat> reference = readAllFrames(path, CAP_GSTREAMER, gstreamerGopFixtureFrameCount);
    ASSERT_EQ((size_t)gstreamerGopFixtureFrameCount, reference.size());

    for (int target = 0; target < gstreamerGopFixtureFrameCount; target++)
    {
        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened());
        cap.set(CAP_PROP_POS_FRAMES, target);
        if (cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)) != 1)
            continue;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
        EXPECT_EQ(target, nearestFrameIndex(frame, reference)) << "target " << target;
    }

    remove(path.c_str());
}

}} // namespace
