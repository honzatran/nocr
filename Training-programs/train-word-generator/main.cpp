
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>
#include <limits>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/iooper.h>
#include <nocrlib/street_view_scene.h>
#include <nocrlib/assert.h>
#include <nocrlib/features.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <boost/program_options.hpp>


#include "extractor.hpp"

#define SIZE 1024
#define ER1_CONF_FILE "../boost_er1stage_handpicked.xml"
#define ER2_CONF_FILE "../scaled_svmEr2_handpicked.xml"


using namespace std;

// string er2_conf_file = "conf/boostGeom.conf";
// const string er2_conf_file = "conf/svmERGeom.conf";


int main( int argc, char **argv )
{
    string line;

    Extractor extractor;
    extractor.loadXml("train_icdar_2013.xml");
    extractor.setUpErTree(ER1_CONF_FILE, ER2_CONF_FILE);

    while (std::getline(cin, line))
    {
        extractor.findImage(line);
    }

    return 0;
}
