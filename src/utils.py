import os
import errno
# import psutil
import numpy as np
import hashlib


def mkdir_p(path):
    """ Equivalent of mkdir -p """
    """ source: http://bit.ly/1dyli3d """
    try:
        os.makedirs(path)
    except OSError as exc:   # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# def memory_usage():
#     """ return the memory usage in MB """
#     """ source: http://bit.ly/1dspz7I """
#     process = psutil.Process(os.getpid())
#     try:
#         mem = process.get_memory_info()[0] / float(2 ** 20)
#     except:
#         mem = process.memory_info()[0] / float(2 ** 20)
#     return mem


def wc_l(fname):
    """ return number of lines in a file """

    line_count = 0
    try:
        with open(fname, 'r') as f:
            for line in f:
                line_count = line_count + 1
    except:
        print('Could not open file ', fname)
        pass
    return line_count


def git_version():
    """ returns git revision """
    """ source: http://bit.ly/1Ctm1ho """

    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()


def todict(obj, classkey=None):
    """ convert object to dict """
    """ source: http://bit.ly/1ZM6Acc  """

    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__"):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
                    for key, value in obj.__dict__.iteritems()
                    if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def save_obj(obj, fname):
    """ save object to npz file """

    obj_dict = todict(obj)
    np.savez(fname, **obj_dict)


def load_obj(obj, fname):
    """ load variables from npz file """

    npzfile = np.load(fname)
    for k in npzfile.files:
        setattr(obj, k, npzfile[k])
    return obj


def sha1_hash(fname, blocksize=4096):
    """ compute sha1hash of a file """
    hash = ''
    if not os.path.exists(fname):
        print "File %s does not exist" % fname
        return ''
    try:
        hasher = hashlib.sha1()
        with open(fname, 'rb') as f:
            buf = f.read(blocksize)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(blocksize)
        hash = hasher.hexdigest()
    except:
        print "Exception in hashing file"
        raise
    return hash
