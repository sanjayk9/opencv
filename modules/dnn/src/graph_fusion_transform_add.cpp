// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Folds `TransformLayout -> NaryEltwise(Add)` into a single fused pass:
//   Block/NHWC -> TransformLayout -> NCHW ---\
//                                              Add -> NCHW result
//                       residual (NCHW) ------/
//                =>
//   Block/NHWC -> TransformLayout(+residual) -> NCHW result
// TransformLayout already deinterleaves the tensor to NCHW; folding the Add
// into that pass's final store saves one full read+write over the tensor.
// Declines whenever the channel count doesn't divide evenly by C0 -- see the
// kernel comment in transform_layout_layer.cpp for why.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "graph_fusion_util.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

struct ModelFusionTransformAdd
{
    explicit ModelFusionTransformAdd(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        size_t nops = prog.size();
        bool modified = false;

        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            vector<Ptr<Graph>>* subgraphs = prog[i]->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseGraph(g)) modified = true;
            }
        }

        vector<int> usecounts;
        netimpl->useCounts(usecounts);

        std::set<int> externalArgs = externalArgSet(graph);
        std::map<int, int> producer = buildProducerMap(prog);

        vector<bool> dropped(nops, false);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            if (!layer || dropped[i]) continue;

            NaryEltwiseLayer* add = dynamic_cast<NaryEltwiseLayer*>(layer.get());
            if (!add) continue;
            if (add->op != NaryEltwiseLayer::OPERATION::ADD &&
                add->op != NaryEltwiseLayer::OPERATION::SUM) continue;
            if (layer->inputs.size() != 2 || layer->outputs.size() != 1) continue;

            for (int slot = 0; slot < 2; slot++) {
                Arg tl_out = layer->inputs[slot];
                auto it = producer.find(tl_out.idx);
                if (it == producer.end()) continue;
                int prod_idx = it->second;
                if (prod_idx < 0 || dropped[prod_idx]) continue;

                const Ptr<LayerInfo>& pl = prog[prod_idx];
                TransformLayoutLayer* tl = dynamic_cast<TransformLayoutLayer*>(pl.get());
                // Already-fused TransformLayout nodes take 2 inputs; a second
                // Add trying to claim the same node is exactly the "used
                // more than once" case the usecount check below also catches,
                // but fusedAdd is checked first since it's cheaper.
                if (!tl || tl->fusedAdd) continue;
                if (tl->layout != DATA_LAYOUT_NCHW) continue;
                if (pl->inputs.size() != 1 || pl->outputs.size() != 1) continue;

                bool single_consumer = usecounts[tl_out.idx] == 1
                                    && externalArgs.count(tl_out.idx) == 0;
                if (!single_consumer) continue;

                // The TransformLayout output and other intermediate results never
                // get a static shape/type recorded on their Arg (inferShapes()/
                // inferTypes() aren't wired up for DNN_ARG_TEMP -- only constants
                // and declared model inputs are populated). So the shape/dtype
                // match can only be checked statically when the residual is one
                // of those two kinds, which is exactly the skip-connection case
                // this pass targets (e.g. high_res_feats_0/1). Anything else is
                // declined here rather than fused on a guess; the exact-shape
                // requirement is still enforced at forward time in
                // TransformLayoutLayerImpl::runOpAdd -> transformLayoutAddF32,
                // which throws instead of silently broadcasting or corrupting.
                Arg residual = layer->inputs[1 - slot];
                ArgKind resKind = netimpl->argData(residual).kind;
                if (resKind != DNN_ARG_CONST && resKind != DNN_ARG_INPUT) continue;

                const ArgData& resData = netimpl->argData(residual);
                if (resData.type != CV_32F) continue;

                // channels() asserts the shape's layout is resolved (BLOCK/
                // NCHW/NHWC); a plain external input's recorded shape can
                // still carry DATA_LAYOUT_UNKNOWN at this point, same as
                // transformLayout() normalizes before using it.
                MatShape resShape = resData.shape;
                if (resShape.layout == DATA_LAYOUT_UNKNOWN)
                    resShape.layout = netimpl->originalLayout;
                int C = resShape.channels();
                if (tl->C0 <= 0 || C <= 0 || C % tl->C0 != 0) continue;

                pl->inputs.push_back(residual);
                tl->fusedAdd = true;
                pl->outputs[0] = layer->outputs[0];
                dropped[i] = true;
                usecounts[tl_out.idx] = 0;
                modified = true;
                break;
            }
        }

        if (modified)
            graph->setProg(rebuildProg(prog, dropped));

        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseTransformLayoutAdd()
{
    if (preferableBackend != DNN_BACKEND_OPENCV ||
        preferableTarget  != DNN_TARGET_CPU)
        return;
    ModelFusionTransformAdd pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
