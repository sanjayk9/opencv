// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple< string, Size, Size, int > Param;
typedef testing::TestWithParam< Param > videoio_gstreamer;

TEST_P(videoio_gstreamer, read_check)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    string format    = get<0>(GetParam());
    Size frame_size  = get<1>(GetParam());
    Size mat_size    = get<2>(GetParam());
    int convertToRGB = get<3>(GetParam());
    int count_frames = 10;
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << count_frames << " ! " << format;
    pipeline << ", width=" << frame_size.width << ", height=" << frame_size.height << " ! appsink";
    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(pipeline.str(), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());

    EXPECT_EQ(CAP_PROP_UNKNOWN, cap.get(CV__CAP_PROP_LATEST));

    Mat buffer, decode_frame, gray_frame, rgb_frame;
    for (int i = 0; i < count_frames; ++i)
    {
        cap >> buffer;
        decode_frame = (format == "jpegenc ! image/jpeg") ? imdecode(buffer, IMREAD_UNCHANGED) : buffer;
        EXPECT_EQ(mat_size, decode_frame.size());

        cvtColor(decode_frame, rgb_frame, convertToRGB);
        cvtColor(rgb_frame, gray_frame, COLOR_RGB2GRAY);
        if (gray_frame.depth() == CV_16U)
        {
            gray_frame.convertTo(gray_frame, CV_8U, 255.0/65535);
        }

        vector<Vec3f> circles;
        HoughCircles(gray_frame, circles, HOUGH_GRADIENT, 1, gray_frame.rows/16, 100, 30, 1, 30 );
        if (circles.size() == 1)
        {
            EXPECT_NEAR(18.5, circles[0][2], 1.0);
        }
        else
        {
            ADD_FAILURE() << "Found " << circles.size() << " on frame " << i ;
        }
    }
    {
        Mat frame;
        cap >> frame;
        EXPECT_TRUE(frame.empty());
    }
    cap.release();
    ASSERT_FALSE(cap.isOpened());
}

static const Param test_data[] = {
    make_tuple("video/x-raw, format=BGR"  , Size(640, 480), Size(640, 480), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=BGRA" , Size(640, 480), Size(640, 480), COLOR_BGRA2RGB),
    make_tuple("video/x-raw, format=RGBA" , Size(640, 480), Size(640, 480), COLOR_RGBA2RGB),
    make_tuple("video/x-raw, format=BGRx" , Size(640, 480), Size(640, 480), COLOR_BGRA2RGB),
    make_tuple("video/x-raw, format=RGBx" , Size(640, 480), Size(640, 480), COLOR_RGBA2RGB),
    make_tuple("video/x-raw, format=GRAY8", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=UYVY" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_UYVY),
    make_tuple("video/x-raw, format=YUY2" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YUY2),
    make_tuple("video/x-raw, format=YVYU" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YVYU),
    make_tuple("video/x-raw, format=NV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV12),
    make_tuple("video/x-raw, format=NV21" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV21),
    make_tuple("video/x-raw, format=YV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_YV12),
    make_tuple("video/x-raw, format=I420" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_I420),
    make_tuple("video/x-bayer"            , Size(640, 480), Size(640, 480), COLOR_BayerBG2RGB),
    make_tuple("jpegenc ! image/jpeg"     , Size(640, 480), Size(640, 480), COLOR_BGR2RGB),

    // unaligned cases, strides information must be used
    make_tuple("video/x-raw, format=BGR"  , Size(322, 242), Size(322, 242), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=GRAY8", Size(322, 242), Size(322, 242), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=NV12" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_NV12),
    make_tuple("video/x-raw, format=NV21" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_NV21),
    make_tuple("video/x-raw, format=YV12" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_YV12),
    make_tuple("video/x-raw, format=I420" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_I420),

    // 16 bit
    make_tuple("video/x-raw, format=GRAY16_LE", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=GRAY16_BE", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
};

INSTANTIATE_TEST_CASE_P(videoio, videoio_gstreamer, testing::ValuesIn(test_data));

TEST(videoio_gstreamer, unsupported_pipeline)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    // could not link videoconvert0 to matroskamux0, matroskamux0 can't handle caps video/x-raw, format=(string)RGBA
    std::string pipeline = "appsrc ! videoconvert ! video/x-raw, format=(string)RGBA ! matroskamux ! filesink location=test.mkv";
    Size frame_size(640, 480);

    VideoWriter writer;
    EXPECT_NO_THROW(writer.open(pipeline, CAP_GSTREAMER, 0/*fourcc*/, 30/*fps*/, frame_size, true));
    EXPECT_FALSE(writer.isOpened());
    // no frames
    EXPECT_NO_THROW(writer.release());

}

TEST(videoio_gstreamer, gray16_writing)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    Size frame_size(320, 240);

    // generate a noise frame
    Mat frame = Mat(frame_size, CV_16U);
    randu(frame, 0, 65535);

    // generate a temp filename, and fix path separators to how GStreamer expects them
    cv::String temp_file = cv::tempfile(".raw");
    std::replace(temp_file.begin(), temp_file.end(), '\\', '/');

    // write noise frame to file using GStreamer
    std::ostringstream writer_pipeline;
    writer_pipeline << "appsrc ! filesink location=" << temp_file;
    std::vector<int> params {
        VIDEOWRITER_PROP_IS_COLOR, 0/*false*/,
        VIDEOWRITER_PROP_DEPTH, CV_16U
    };
    VideoWriter writer;
    ASSERT_NO_THROW(writer.open(writer_pipeline.str(), CAP_GSTREAMER, 0/*fourcc*/, 30/*fps*/, frame_size, params));
    ASSERT_TRUE(writer.isOpened());
    ASSERT_NO_THROW(writer.write(frame));
    ASSERT_NO_THROW(writer.release());

    // read noise frame back in
    Mat written_frame(frame_size, CV_16U);
    std::ifstream fs(temp_file, std::ios::in | std::ios::binary);
    fs.read((char*)written_frame.ptr(0), frame_size.width * frame_size.height * 2);
    ASSERT_TRUE(fs);
    fs.close();

    // compare to make sure it's identical
    EXPECT_EQ(0, cv::norm(frame, written_frame, NORM_INF));

    // remove temp file
    EXPECT_EQ(0, remove(temp_file.c_str()));
}

TEST(videoio_gstreamer, timeout_property)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    cap.open("videotestsrc ! appsink", CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());
    const double default_timeout = 30000; // 30 seconds
    const double open_timeout = 5678; // 3 seconds
    const double read_timeout = 1234; // 1 second
    EXPECT_NEAR(default_timeout, cap.get(CAP_PROP_OPEN_TIMEOUT_MSEC), 1e-3);
    const double current_read_timeout = cap.get(CAP_PROP_READ_TIMEOUT_MSEC);
    const bool read_timeout_supported = current_read_timeout > 0.0;
    if (read_timeout_supported)
    {
        EXPECT_NEAR(default_timeout, current_read_timeout, 1e-3);
    }
    cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout);
    EXPECT_NEAR(open_timeout, cap.get(CAP_PROP_OPEN_TIMEOUT_MSEC), 1e-3);
    if (read_timeout_supported)
    {
        cap.set(CAP_PROP_READ_TIMEOUT_MSEC, read_timeout);
        EXPECT_NEAR(read_timeout, cap.get(CAP_PROP_READ_TIMEOUT_MSEC), 1e-3);
    }
}

//==============================================================================
// Seeking test with manual GStreamer pipeline
typedef testing::TestWithParam<string> gstreamer_bunny;

TEST_P(gstreamer_bunny, manual_seek)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const string video_file = BunnyParameters::getFilename("." + GetParam());
    const string pipeline = "filesrc location=" + video_file + " ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
    const double target_pos = 3000.0;
    const double ms_per_frame = 1000.0 / BunnyParameters::getFps();
    VideoCapture cap;
    cap.open(pipeline, CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());
    Mat img;
    for (int i = 0; i < 10; i++)
    {
        cap >> img;
    }
    EXPECT_FALSE(img.empty());
    cap.set(CAP_PROP_POS_MSEC, target_pos);
    cap >> img;
    EXPECT_FALSE(img.empty());
    double actual_pos = cap.get(CAP_PROP_POS_MSEC);
    EXPECT_NEAR(actual_pos, target_pos, ms_per_frame);
}

static const string bunny_params[] = {
    // string("wmv"),
    string("mov"),
    string("mp4"),
    // string("mpg"),
    string("avi"),
    // string("h264"),
    // string("h265"),
    string("mjpg.avi")
};

inline static std::string gstreamer_bunny_name_printer(const testing::TestParamInfo<gstreamer_bunny::ParamType>& info)
{
    std::ostringstream out;
    out << extToStringSafe(info.param);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(videoio, gstreamer_bunny, testing::ValuesIn(bunny_params), gstreamer_bunny_name_printer);

static std::string gstEncoderPipeline(const std::string& encoder, const std::string& file)
{
    return "appsrc ! videoconvert ! " + encoder + " ! matroskamux ! filesink location=" + file;
}

static bool gstEncoderAvailable(const std::string& encoder)
{
    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    const bool ok = writer.open(gstEncoderPipeline(encoder, file), CAP_GSTREAMER, 0, 25, Size(320, 240));
    writer.release();
    remove(file.c_str());
    return ok;
}

static void generateGstFrames(std::vector<Mat>& frames, Size size, int count, bool staticBackground = false)
{
    frames.clear();
    RNG& rng = theRNG();
    Mat background(size, CV_8UC3);
    rng.fill(background, RNG::UNIFORM, 0, 255);
    for (int i = 0; i < count; i++)
    {
        Mat frame(size, CV_8UC3);
        if (staticBackground)
            background.copyTo(frame);
        else
            rng.fill(frame, RNG::UNIFORM, 0, 255);
        circle(frame, Point((i * 13) % size.width, (i * 7) % size.height), 40, Scalar::all(255), -1);
        frames.push_back(frame);
    }
}

static void writeGstFrames(VideoWriter& writer, const std::vector<Mat>& frames)
{
    for (size_t i = 0; i < frames.size(); i++)
        writer.write(frames[i]);
}

static long fileSize(const std::string& file)
{
    std::ifstream fs(file.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return fs ? (long)fs.tellg() : -1;
}

static long writeWithGstEncoderParams(const std::string& file, const std::vector<int>& params,
                                      const std::vector<Mat>& frames, double* readBack = NULL, int prop = -1)
{
    VideoWriter writer;
    if (!writer.open(gstEncoderPipeline("x264enc", file), CAP_GSTREAMER, 0, 25, frames[0].size(), params))
        return -1;
    if (readBack && prop >= 0)
        *readBack = writer.get(prop);
    writeGstFrames(writer, frames);
    writer.release();
    return fileSize(file);
}

static int maxKeyFrameGap(const std::string& file)
{
    VideoCapture cap(file, CAP_FFMPEG, {CAP_PROP_FORMAT, -1});
    if (!cap.isOpened())
        return -1;
    int gap = 0, maxGap = 0;
    bool seenKeyFrame = false;
    while (cap.grab())
    {
        if (cap.get(CAP_PROP_LRF_HAS_KEY_FRAME) != 0)
        {
            maxGap = std::max(maxGap, gap);
            gap = 0;
            seenKeyFrame = true;
        }
        gap++;
    }
    return seenKeyFrame ? std::max(maxGap, gap) : -1;
}

TEST(videoio_gstreamer_encoder_props, bitrate_changes_size)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    std::vector<Mat> frames;
    generateGstFrames(frames, Size(320, 240), 30);

    const string lowFile = cv::tempfile(".mkv");
    const string highFile = cv::tempfile(".mkv");
    double readBack = -1;
    const long lowSize = writeWithGstEncoderParams(lowFile, {VIDEOWRITER_PROP_BITRATE, 200000}, frames,
                                                   &readBack, VIDEOWRITER_PROP_BITRATE);
    const long highSize = writeWithGstEncoderParams(highFile, {VIDEOWRITER_PROP_BITRATE, 4000000}, frames);
    ASSERT_GT(lowSize, 0);
    ASSERT_GT(highSize, 0);
    EXPECT_EQ(200000, (int)readBack);
    EXPECT_GT(highSize, lowSize);
    remove(lowFile.c_str());
    remove(highFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, crf_changes_size)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    std::vector<Mat> frames;
    generateGstFrames(frames, Size(320, 240), 30);

    const string lowFile = cv::tempfile(".mkv");
    const string highFile = cv::tempfile(".mkv");
    double readBack = -1;
    const long lowSize = writeWithGstEncoderParams(lowFile, {VIDEOWRITER_PROP_CRF, 18}, frames,
                                                   &readBack, VIDEOWRITER_PROP_CRF);
    const long highSize = writeWithGstEncoderParams(highFile, {VIDEOWRITER_PROP_CRF, 40}, frames);
    ASSERT_GT(lowSize, 0);
    ASSERT_GT(highSize, 0);
    EXPECT_EQ(18, (int)readBack);
    EXPECT_GT(lowSize, highSize);
    remove(lowFile.c_str());
    remove(highFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, gop_limits_key_frame_interval)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend is needed to read key frames");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    std::vector<Mat> frames;
    generateGstFrames(frames, Size(320, 240), 30, true);

    const string defaultFile = cv::tempfile(".mkv");
    const string gopFile = cv::tempfile(".mkv");
    double readBack = -1;
    ASSERT_GT(writeWithGstEncoderParams(defaultFile, std::vector<int>(), frames), 0);
    ASSERT_GT(writeWithGstEncoderParams(gopFile, {VIDEOWRITER_PROP_GOP_SIZE, 5}, frames,
                                        &readBack, VIDEOWRITER_PROP_GOP_SIZE), 0);
    EXPECT_EQ(5, (int)readBack);
    EXPECT_GT(maxKeyFrameGap(defaultFile), 5);
    const int gap = maxKeyFrameGap(gopFile);
    EXPECT_GT(gap, 0);
    EXPECT_LE(gap, 5);
    remove(defaultFile.c_str());
    remove(gopFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, preset_round_trip)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    std::vector<Mat> frames;
    generateGstFrames(frames, Size(320, 240), 20);

    const string file = cv::tempfile(".mkv");
    double readBack = -1;
    ASSERT_GT(writeWithGstEncoderParams(file, {VIDEOWRITER_PROP_PRESET, VIDEOWRITER_PRESET_ULTRAFAST}, frames,
                                        &readBack, VIDEOWRITER_PROP_PRESET), 0);
    EXPECT_EQ(VIDEOWRITER_PRESET_ULTRAFAST, (int)readBack);
    remove(file.c_str());
}

TEST(videoio_gstreamer_encoder_props, unsupported_property_fails_open)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("avenc_mpeg4"))
        throw SkipTestException("avenc_mpeg4 is not available");

    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    EXPECT_FALSE(writer.open(gstEncoderPipeline("avenc_mpeg4", file), CAP_GSTREAMER, 0, 25, Size(320, 240),
                             {VIDEOWRITER_PROP_PRESET, VIDEOWRITER_PRESET_VERYSLOW}));
    remove(file.c_str());
}

TEST(videoio_gstreamer_encoder_props, fourcc_path_encodes)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const Size size(320, 240);
    const int count = 20;
    std::vector<Mat> frames;
    generateGstFrames(frames, size, count, true);

    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    ASSERT_TRUE(writer.open(file, CAP_GSTREAMER, VideoWriter::fourcc('X', '2', '6', '4'), 25, size));
    writeGstFrames(writer, frames);
    writer.release();

    const long rawSize = (long)size.area() * 3 / 2 * count;
    const long encodedSize = fileSize(file);
    ASSERT_GT(encodedSize, 0);
    EXPECT_LT(encodedSize, rawSize / 4);
    remove(file.c_str());

    const string nv12File = cv::tempfile(".mkv");
    VideoWriter nv12Writer;
    ASSERT_TRUE(nv12Writer.open(nv12File, CAP_GSTREAMER, VideoWriter::fourcc('X', '2', '6', '4'), 25, size,
                                {VIDEOWRITER_PROP_COLOR_SPACE, VideoWriter::fourcc('N', 'V', '1', '2')}));
    writeGstFrames(nv12Writer, frames);
    nv12Writer.release();
    const long nv12Size = fileSize(nv12File);
    ASSERT_GT(nv12Size, 0);
    EXPECT_LT(nv12Size, rawSize / 4);
    remove(nv12File.c_str());
}
//==============================================================================
// CAP_PROP_POS_FRAMES seeks must land on the requested frame

static std::vector<Mat> readAllGstreamerFrames(const std::string& source, int maxFrames)
{
    std::vector<Mat> frames;
    VideoCapture cap(source, CAP_GSTREAMER);
    Mat frame;
    while ((int)frames.size() < maxFrames && cap.read(frame))
        frames.push_back(frame.clone());
    return frames;
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

static const int gopFixtureFrameCount = 30;

// Each encoded frame is a flat, distinct color.
static Scalar gopFixtureFrameColor(int i)
{
    return Scalar(i * 5 % 256, (i * 7 + 30) % 256, (i * 11 + 60) % 256);
}

// By default an mp4 with a key frame every 12 frames, so most seek targets sit between key frames.
static std::string generateGopFixture(const std::string& x264Options = "key-int-max=12 bframes=0",
                                      const std::string& muxer = "qtmux", const std::string& ext = ".mp4")
{
    std::string path = cv::tempfile(ext.c_str());
    std::string pipeline = "appsrc ! videoconvert ! x264enc " + x264Options + " ! h264parse ! " + muxer + " ! filesink location=" + path;
    VideoWriter writer(pipeline, CAP_GSTREAMER, 0, 24.0, Size(64, 48), true);
    if (!writer.isOpened())
        return std::string();
    for (int i = 0; i < gopFixtureFrameCount; i++)
        writer.write(Mat(48, 64, CV_8UC3, gopFixtureFrameColor(i)));
    return path;
}

static std::string posFramesPipelineWithoutDuration(int srcFrameCount, double srcFps)
{
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << srcFrameCount
             << " ! video/x-raw,framerate=" << cvRound(srcFps) << "/1 ! appsink";
    return pipeline.str();
}

// videotestsrc has no duration: only a seek to frame 0 works, and a failed seek must not claim exact.
TEST(videoio_gstreamer_pos_frames, pipeline_without_duration_only_seeks_to_start)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(posFramesPipelineWithoutDuration(30, 30.0), CAP_GSTREAMER));
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

TEST(videoio_gstreamer_pos_frames, real_file_seek_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap(findDataFile("video/big_buck_bunny.mp4"), CAP_GSTREAMER);
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

// Seeks back and forth on one capture must report the requested frame and read back its content.
TEST(videoio_gstreamer_pos_frames, seek_lands_on_the_requested_frame)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGopFixture();
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    std::vector<Mat> reference = readAllGstreamerFrames(path, gopFixtureFrameCount + 1);
    ASSERT_EQ((size_t)gopFixtureFrameCount, reference.size());

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
TEST(videoio_gstreamer_pos_frames, seek_to_every_frame_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    for (const char* name : {"video/rotated_metadata.mp4", "video/big_buck_bunny.mp4"})
    {
        const std::string path = findDataFile(name);
        std::vector<Mat> reference = readAllGstreamerFrames(path, 200);
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
TEST(videoio_gstreamer_pos_frames, open_gop_seek_does_not_claim_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGopFixture("key-int-max=12 bframes=3 option-string=open-gop=1");
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    std::vector<Mat> reference = readAllGstreamerFrames(path, gopFixtureFrameCount);
    ASSERT_EQ((size_t)gopFixtureFrameCount, reference.size());

    int exactCount = 0;
    for (int target = 0; target < gopFixtureFrameCount; target++)
    {
        VideoCapture cap(path, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened());
        cap.set(CAP_PROP_POS_FRAMES, target);
        if (cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)) != 1)
            continue;
        exactCount++;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
        EXPECT_EQ(target, nearestFrameIndex(frame, reference)) << "target " << target;
    }
    // Without this the loop could skip every target and pass having checked nothing.
    EXPECT_GT(exactCount, 0);

    remove(path.c_str());
}

// Manual demuxer pipelines ending in appsink drop=1 (#24243) must still seek exactly from the PAUSED state.
TEST(videoio_gstreamer_pos_frames, manual_demux_pipeline_seek_is_exact)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string fixture = generateGopFixture();
    if (fixture.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    for (const std::string& path : {fixture, findDataFile("video/big_buck_bunny.mp4"), findDataFile("video/big_buck_bunny.mov")})
    {
        const std::string pipeline = "filesrc location=" + path + " ! qtdemux name=demux demux.video_0 ! decodebin"
                                     " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
        // Reference from a plain file capture: the manual pipeline's appsink syncs to the clock and decodes in real time.
        std::vector<Mat> reference = readAllGstreamerFrames(path, gopFixtureFrameCount);
        ASSERT_EQ((size_t)gopFixtureFrameCount, reference.size()) << path;

        VideoCapture cap(pipeline, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened()) << path;
        // Read first so the pipeline is PLAYING, the state #24243 stopped the pipeline from.
        Mat frame;
        for (int i = 0; i < 3; i++)
            ASSERT_TRUE(cap.read(frame)) << path;

        for (int target : {17, 5, 22, 0, 13, 29, 8})
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << path << " target " << target;
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << path << " target " << target;
            EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << path << " target " << target;
            ASSERT_TRUE(cap.read(frame)) << path << " target " << target;
            EXPECT_EQ(target, nearestFrameIndex(frame, reference)) << path << " target " << target;
        }
    }
    remove(fixture.c_str());
}

// matroskademux pipelines report a non-zero position at open, so POS_FRAMES is disabled; set() must refuse without claiming exact.
TEST(videoio_gstreamer_pos_frames, manual_matroska_pipeline_refuses_seek)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    std::string path = generateGopFixture("key-int-max=12 bframes=0", "matroskamux", ".mkv");
    if (path.empty())
        throw SkipTestException("GStreamer x264enc encoder was not available to build the test fixture");

    {
        const std::string pipeline = "filesrc location=" + path + " ! matroskademux name=demux demux.video_0 ! decodebin"
                                     " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
        VideoCapture cap(pipeline, CAP_GSTREAMER);
        ASSERT_TRUE(cap.isOpened());
        Mat frame;
        for (int i = 0; i < 3; i++)
            ASSERT_TRUE(cap.read(frame));

        EXPECT_FALSE(cap.set(CAP_PROP_POS_FRAMES, 17));
        EXPECT_EQ(-1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT)));
        EXPECT_TRUE(cap.read(frame));
    }
    remove(path.c_str());
}

}} // namespace
