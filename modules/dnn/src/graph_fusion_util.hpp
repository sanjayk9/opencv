// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Shared scaffolding for the single-pass structural fusion passes in
// graph_fusion_*.cpp: build a producer (Arg -> op index) map, the set of
// Args that cross the graph boundary, and rebuild a program after dropping
// absorbed layers. Each pass still owns its own subgraph recursion and
// use-count refresh, since those interleave with pass-specific matching
// logic in ways that don't factor out as cleanly as these three.

#ifndef OPENCV_DNN_GRAPH_FUSION_UTIL_HPP
#define OPENCV_DNN_GRAPH_FUSION_UTIL_HPP

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Maps each Arg produced anywhere in `prog` to the index of the op that produces it.
inline std::map<int, int> buildProducerMap(const std::vector<Ptr<LayerInfo>>& prog)
{
    std::map<int, int> producer;
    for (size_t i = 0; i < prog.size(); i++) {
        if (!prog[i]) continue;
        for (Arg out : prog[i]->outputs)
            producer[out.idx] = (int)i;
    }
    return producer;
}

// Args that are declared outputs of `graph` and so must not be treated as a
// purely-internal, single-use intermediate by a fusion pass.
inline std::set<int> externalArgSet(const Ptr<Graph>& graph)
{
    std::set<int> externalArgs;
    for (Arg out : graph->outputs())
        externalArgs.insert(out.idx);
    return externalArgs;
}

// Rebuilds `prog` with every op marked in `dropped` removed, preserving order.
inline std::vector<Ptr<LayerInfo>> rebuildProg(const std::vector<Ptr<LayerInfo>>& prog,
                                                const std::vector<bool>& dropped)
{
    std::vector<Ptr<LayerInfo>> newprog;
    newprog.reserve(prog.size());
    for (size_t i = 0; i < prog.size(); i++) {
        if (!dropped[i] && prog[i])
            newprog.push_back(prog[i]);
    }
    return newprog;
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
