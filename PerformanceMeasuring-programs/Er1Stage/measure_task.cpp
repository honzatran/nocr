
#include "measure_task.hpp"


ErTreeBuildTask::ErTreeBuildTask()
{
    er_tree.setMinAreaRatio(k_min_area_ratio);
    er_tree.setMaxAreaRatio(k_max_area_ratio);
    er_tree.setMinGlobalProbability(0.2);
    er_tree.setMinDifference(0.1);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(k_er1_conf_file);

    er_tree.setERFunction(std::move(er_function));
}
