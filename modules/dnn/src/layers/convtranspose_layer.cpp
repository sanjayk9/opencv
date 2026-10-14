// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "cpu_kernels/mlas_gemm.hpp"

namespace cv
{
namespace dnn
{

/*
    ONNX Det operator
    Spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    Supported opsets: 1-22
*/

class ConvTranspose2LayerImpl : public ConvTranspose2Layer
{
public:
    ConvTranspose2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        adjust_pads = params.getVector<int>("adj");
        // ONNX output_shape (spatial only); fixes the output, paddings derived from it.
        explicit_out_shape = params.getVector<int>("output_shape_spatial");
        ngroups = params.get<int>("group", 1);
    }

    void applyExplicitOutShape(MatShape& outshape) const
    {
        if (explicit_out_shape.empty())
            return;
        int nsd = outshape.dims - 2 - int(outshape.layout == DATA_LAYOUT_BLOCK);
        CV_CheckEQ((int)explicit_out_shape.size(), nsd, "output_shape must cover all spatial dims");
        for (int i = 0; i < nsd; i++)
            outshape[i + 2] = explicit_out_shape[i];
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "group: " << ngroups << ",\n";

        if (!wshape0.empty()) {
            prindent(strm, indent);
            strm << "ksize: [";
            for (int k = 0; k < wshape0.dims; k++)
                strm << (k > 0 ? ", " : "") << wshape0[k];
            strm << "],\n";
        }

        prindent(strm, indent);
        strm << "stride: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << strides[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "dilation: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pad: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << pads[k];
        strm << "],\n";

        if (!adjust_pads.empty()) {
            prindent(strm, indent);
            strm << "adj: [";
            for (size_t k = 0; k < adjust_pads.size(); k++)
                strm << (k > 0 ? ", " : "") << adjust_pads[k];
            strm << "],\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void setWeights(InputArray weights_arr, InputArray bias_arr,
                            int C0, int /*accuracy*/) CV_OVERRIDE
    {
        Mat rawWeights = weights_arr.getMat();
        Mat rawBias = bias_arr.getMat();
        CV_Assert(!rawWeights.empty());

        wshape0 = rawWeights.shape();

        Mat wfloat;
        if (rawWeights.type() != CV_32F)
            rawWeights.convertTo(wfloat, CV_32F);
        else
            wfloat = rawWeights;
        repackDeconvWeights(wfloat, weights, CV_32F, ngroups, C0);
        packGemmWeights(wfloat, C0);

        if (!rawBias.empty())
            rawBias.convertTo(bias, CV_32F);
    }

    // With kernel == stride, dilation 1 and one group, every output pixel gets exactly one tap, so the
    // layer is [pixels x Cin] * [Cin x (taps, Cout padded to C0)] followed by a scatter (forwardGemm).
    void packGemmWeights(const Mat& w, int C0)
    {
        gemmWeights.release();
        gemmN = gemmK = gemmC0 = 0;
        if (ngroups != 1 || !mlasAvailable() || !w.isContinuous())
            return;
        int nsd = w.dims - 2;
        for (int i = 0; i < nsd; i++) {
            int k = w.size[2 + i];
            int s = strides.empty() ? 1 : strides[i];
            int d = dilations.empty() ? 1 : dilations[i];
            if (k != s || (k > 1 && d != 1))
                return;
        }
        const int Cin = w.size[0], Cout = w.size[1];
        const int ksize = (int)(w.total() / ((size_t)Cin * Cout));
        const int Kpad = (Cout + C0 - 1) / C0 * C0;
        const int N = ksize * Kpad;

        Mat B(Cin, N, CV_32F, Scalar(0));
        const float* wdata = w.ptr<float>();
        for (int c = 0; c < Cin; c++) {
            float* brow = B.ptr<float>(c);
            for (int k = 0; k < Cout; k++) {
                const float* wk = wdata + ((size_t)c * Cout + k) * ksize;
                for (int t = 0; t < ksize; t++)
                    brow[t * Kpad + k] = wk[t];
            }
        }

        size_t bytes = mlasSgemmPackBSize(false, false, N, Cin);
        if (bytes == 0 || bytes > (size_t)INT_MAX)
            return;
        gemmWeights.create(1, (int)bytes, CV_8U);
        if (!mlasSgemmPackB(false, false, N, Cin, B.ptr<float>(), N, gemmWeights.data)) {
            gemmWeights.release();
            return;
        }
        gemmN = N;
        gemmK = Cin;
        gemmC0 = C0;
    }

    // Output is exactly input * stride (no padding, no output_padding), so taps tile it.
    bool gemmCoversShape(const MatShape& inpshape, const MatShape& outshape) const
    {
        if (gemmWeights.empty() || inpshape.C != gemmK || inpshape.back() != gemmC0 ||
            outshape.back() != gemmC0 || gemmN % (outshape[1] * gemmC0) != 0)
            return false;
        const int sdims = cs.nspatialdims;
        for (int i = 0; i < sdims; i++) {
            int j = ConvState::MAX_CONV_DIMS - sdims + i;
            if (cs.kshape[j] != cs.strides[j] || cs.pads[j] != 0 ||
                cs.pads[j + ConvState::MAX_CONV_DIMS] != 0 ||
                outshape[2 + i] != inpshape[2 + i] * cs.strides[j])
                return false;
        }
        return true;
    }

    void forwardGemm(const Mat& inp, Mat& out)
    {
        enum { MAX_DIMS = ConvState::MAX_CONV_DIMS };
        const MatShape& inpshape = inp.shape();
        const MatShape& outshape = out.shape();
        const int sdims = cs.nspatialdims, C0 = inpshape.back();
        const int N = inpshape[0], C1 = inpshape[1], K1 = outshape[1];
        const int Cin = gemmK, ncols = gemmN, Kpad = K1 * C0, ksize = ncols / Kpad;
        CV_Assert(ksize * Kpad == ncols);

        int idims[MAX_DIMS], ostep[MAX_DIMS];
        int ispatial = 1, ospatial = 1;
        for (int i = sdims - 1; i >= 0; i--) {
            idims[i] = inpshape[2 + i];
            ostep[i] = ospatial;
            ispatial *= inpshape[2 + i];
            ospatial *= outshape[2 + i];
        }

        // Output offset of each tap relative to the tap-0 pixel of its input position.
        std::vector<int> tapofs(ksize);
        for (int t = 0; t < ksize; t++) {
            int tmp = t, ofs = 0;
            for (int i = sdims - 1; i >= 0; i--) {
                int k = cs.kshape[MAX_DIMS - sdims + i];
                ofs += (tmp % k) * ostep[i];
                tmp /= k;
            }
            tapofs[t] = ofs;
        }

        std::vector<float> biasbuf(Kpad, 0.f);
        if (!bias.empty())
            std::copy(bias.ptr<float>(), bias.ptr<float>() + std::min((int)bias.total(), Kpad),
                      biasbuf.begin());

        // A few row blocks per thread for balance; a block's rows stay in cache across the
        // gather, the GEMM and the scatter. The GEMM runs serially inside each task.
        const int nthreads = std::max(getNumThreads(), 1);
        const int rows = (int)std::min<int64_t>(256, std::max<int64_t>(16,
                             (int64_t)N * ispatial / (nthreads * 4)));
        const int nblocks = (ispatial + rows - 1) / rows;

        const float* inpdata = inp.ptr<float>();
        float* outdata = out.ptr<float>();
        const float* biasdata = biasbuf.data();

        parallel_for_(Range(0, N * nblocks), [&](const Range& range) {
            AutoBuffer<float> buf((size_t)rows * (Cin + ncols));
            float* A = buf.data();
            float* Cm = A + (size_t)rows * Cin;

            for (int task = range.start; task < range.end; task++) {
                const int n = task / nblocks;
                const int p0 = (task % nblocks) * rows;
                const int M = std::min(rows, ispatial - p0);
                const float* inp_n = inpdata + (size_t)n * C1 * ispatial * C0;
                float* out_n = outdata + (size_t)n * K1 * ospatial * C0;

                for (int m = 0; m < M; m++) {
                    float* arow = A + (size_t)m * Cin;
                    for (int c1 = 0; c1 < C1; c1++) {
                        const float* src = inp_n + ((size_t)c1 * ispatial + p0 + m) * C0;
                        int nc = std::min(C0, Cin - c1 * C0);
                        std::copy(src, src + nc, arow + c1 * C0);
                    }
                }

                bool ok = mlasSgemmPacked(false, false, M, ncols, Cin, 1.f, A, Cin,
                                          gemmWeights.data, 0.f, Cm, ncols);
                CV_Assert(ok);

                for (int m = 0; m < M; m++) {
                    int tmp = p0 + m, obase = 0;
                    for (int i = sdims - 1; i >= 0; i--) {
                        obase += (tmp % idims[i]) * cs.strides[MAX_DIMS - sdims + i] * ostep[i];
                        tmp /= idims[i];
                    }
                    const float* crow = Cm + (size_t)m * ncols;
                    for (int t = 0; t < ksize; t++) {
                        const int opos = obase + tapofs[t];
                        const float* src = crow + t * Kpad;
                        for (int k1 = 0; k1 < K1; k1++) {
                            float* dst = out_n + ((size_t)k1 * ospatial + opos) * C0;
                            const float* s = src + k1 * C0;
                            const float* b = biasdata + k1 * C0;
                            for (int j = 0; j < C0; j++)
                                dst[j] = s[j] + b[j];
                        }
                    }
                }
            }
        });
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1)
            return false;
        Mat new_bias = arr.getMat();
        CV_Assert(new_bias.isContinuous() && new_bias.dims == 1);
        if (new_bias.type() != CV_32F) {
            Mat temp;
            new_bias.convertTo(temp, CV_32F);
            new_bias = temp;
        }
        if (!bias.empty()) {
            CV_Assert(bias.shape() == new_bias.shape());
            add(bias, new_bias, bias);
        } else {
            new_bias.copyTo(bias);
        }
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape>& inputs,
                             const std::vector<MatShape>& outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        MatShape wshape = inputs.size() > 1 ? inputs[1] : wshape0;
        int K = wshape[1] * ngroups;
        size_t ksize = wshape.total() / (wshape[0] * wshape[1]);
        int C = inputs[0][1] * inputs[0].back();
        return (int64_t)((inputs[0].total() / C) * ksize * K);
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs >= 1);
        outtypes.assign(1, inferType(inptypes[0]));
        temptypes.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape>& outshapes,
                                 std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, deconvInferShape(inpshapes[0], wshape, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad));
        applyExplicitOutShape(outshapes[0]);
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u && requiredOutputs == 1u);
        desiredInputs = actualInputs;
        desiredInputs[0] = DATA_LAYOUT_BLOCK;
        for (size_t i = 1; i < ninputs; i++)
            desiredInputs[i] = DATA_LAYOUT_UNKNOWN;
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = input_arrs.getMat(0);
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        bool dynamicWeights = false;
        for (int i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                dynamicWeights = true;
        }
        if (dynamicWeights || weights.empty()) {
            setWeights(input_arrs.getMat(1), ninputs > 2 ? input_arrs.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = deconvInferShape(inpshape, wshape0, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad);
        applyExplicitOutShape(outshape);

        // compute actual pads for SAME/VALID auto-padding, or from an explicit output_shape
        int nsd = inpshape.dims - 3;
        std::vector<int> pads_resolved = pads;
        if (auto_pad != AUTO_PAD_NONE || !explicit_out_shape.empty()) {
            pads_resolved.resize(nsd * 2, 0);
            for (int i = 0; i < nsd; i++) {
                int inpsz   = inpshape[2 + i];
                int outsz   = outshape[2 + i];
                int adj_i   = adjust_pads.empty() ? 0 : adjust_pads[i];
                int stride  = strides.empty() ? 1 : strides[i];
                int dil     = dilations.empty() ? 1 : dilations[i];
                int ki      = wshape0[2 + i];
                int total   = (inpsz - 1) * stride + dil * (ki - 1) + 1 + adj_i - outsz;
                total = std::max(total, 0);
                // ONNX splits total_padding evenly; odd pixel goes last for SAME_UPPER.
                int pb = (auto_pad == AUTO_PAD_SAME_UPPER) ? total / 2 : total - total / 2;
                pads_resolved[i]       = pb;
                pads_resolved[nsd + i] = total - pb;
            }
        }

        int outtype = inferType(inptype);

        if (inpshape != prevInpshape) {
            cs.initDeconv(inpshape, wshape0, outshape, ngroups,
                          strides, dilations, pads_resolved);
            prevInpshape = inpshape;
        }

        int outkind = output_arrs.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

        std::vector<Mat>* outs = nullptr;
        std::vector<UMat>* uouts = nullptr;
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            outs = &output_arrs.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &output_arrs.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        if (inptype == CV_32F && gemmCoversShape(inpshape, outshape)) {
            forwardGemm(inp, out);
        } else {
            DeconvFunc func = getDeconvFunc(inptype);
            CV_Assert(func != nullptr);
            const float* bias_data = bias.empty() ? nullptr : bias.ptr<float>();
            func(inp.data, nullptr, out.data, cs, weights.data, nullptr, bias_data);
        }

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamicWeights) {
            weights.release();
            gemmWeights.release();
        }
    }

    std::vector<int> emptyKernelShape;
    std::vector<int> explicit_out_shape;
    Mat weights, bias;
    Mat gemmWeights;  // MLAS-packed B for forwardGemm(), empty when the shape doesn't qualify
    int gemmN = 0, gemmK = 0, gemmC0 = 0;
    MatShape wshape0, prevInpshape;
    ConvState cs;
};

Ptr<ConvTranspose2Layer> ConvTranspose2Layer::create(const LayerParams& params)
{
    return Ptr<ConvTranspose2Layer>(new ConvTranspose2LayerImpl(params));
}

}}
