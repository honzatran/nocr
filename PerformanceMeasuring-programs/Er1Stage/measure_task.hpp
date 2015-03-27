

#ifndef _MEASURE_TASK
#define _MEASURE_TASK

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>

#include <vector>

typedef unsigned long long uUINT64;



#if _MSC_VER

#else
#include <sys/time.h>

inline uUINT64 getRelativeTime()
{
    ::timeval tv;

    if (gettimeofday(&tv, 0))
    {
        return 0;
    }

    return ((uUINT64)tv.tv_sec * 1000) + tv.tv_usec/1000;
}

#endif

template <typename G, typename D> 
struct RunTask
{
    static void run(G & task, const D & data);
};

template <typename G, typename D, int ITER> 
class MeasureTask
{
public:
    double measureWallClockTime(G & task, const D & data)
    {
        uUINT64 start = getRelativeTime();
        for (int i = 0; i < ITER; ++i)
        {
            RunTask<G,D>::run(task, data);
        }

        uUINT64 end = getRelativeTime();

        return (double)(end - start)/ITER;
    }
};




struct ErTreeBuildTask
{
    ErTreeBuildTask();

    ERTree er_tree;

    const double k_min_area_ratio = 0.00007;
    const double k_max_area_ratio = 0.3;
    const string k_er1_conf_file = "../boost_er1stage.conf";
};

template <> 
struct RunTask<ErTreeBuildTask, std::vector<cv::Mat> > 
{
    static void run(
            ErTreeBuildTask & task, 
            const std::vector<cv::Mat> & data)
    {
        for (const auto & image : data)
        {
            task.er_tree.setImage(image);
            ComponentTreeBuilder<ERTree> builder( &task.er_tree );
            builder.buildTree();
            task.er_tree.transformExtreme();
            task.er_tree.rejectSimilar();
            task.er_tree.invertDomain();
            auto comps = task.er_tree.toComponent();
            task.er_tree.deallocateTree();
            builder.buildTree();
            task.er_tree.transformExtreme();
            task.er_tree.rejectSimilar();
            auto tmp = task.er_tree.toComponent();
            comps.insert(comps.end(), tmp.begin(), tmp.end());
            task.er_tree.deallocateTree();
        }
    }
};



#endif
