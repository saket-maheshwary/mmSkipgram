from __future__ import print_function
import time
import pdb
import logging
import sys
import platform
from datetime import datetime
import utils
from compute_cnn_features import compute_cnn_features


if __name__ == '__main__':
    runinfo = {}
    runinfo['platform'] = platform.uname()
    runinfo['git_rev'] = utils.git_version()
    runinfo['start_time'] = datetime.now()
    runinfo['end_time'] = None
    runinfo['params'] = {}
    print('Platform: ', runinfo['platform'])
    print('Git revision:', runinfo['git_rev'])
    print('Start Time: ', runinfo['start_time'])
    start_time = time.time()
    start_ctime = time.clock()
    try:

        compute_cnn_features()

        end_time = time.time()
        end_ctime = time.clock()
        print('Success: wall time:  %f sec, processor time: %f sec'
              % (end_time - start_time, end_ctime - start_ctime))
        runinfo['end_time'] = datetime.now()
        print('End Time: ', runinfo['end_time'])
        sys.stdout.flush()
    except:
        end_time = time.time()
        end_ctime = time.clock()
        print('Failure: wall time: %f sec, processor time: %f sec' %
              (end_time - start_time, end_ctime - start_ctime))
        runinfo['end_time'] = datetime.now()
        print('End Time: ', runinfo['end_time'])
        logging.exception(sys.exc_info()[2])
        pdb.post_mortem(sys.exc_info()[2])
