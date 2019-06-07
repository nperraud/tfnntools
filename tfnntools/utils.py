import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle


import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile


def arg_helper(params, d_params):
    """Check if all parameter of d_params are in params. If not, they are added to params."""
    for key in d_params.keys():
        params[key] = params.get(key, d_params[key])
        if isdict(params[key])  and isdict(d_params[key]):
            params[key] = arg_helper(params[key], d_params[key])
    check_keys(params, d_params)
    return params

def check_keys(params, d_params, upperkeys = ''):
    """Check recursively if params and d_params if all the keys of params are in d_params."""
    keys = set(d_params.keys())
    for key in params.keys():
        if key not in keys:
            print('Warning! Optional argument: {}[\'{}\'] specified by user but not used'.format(upperkeys,key))
        else:
            if isdict(params[key]):
#                 if not(isdict(d_params[key])):
#                     print('Warning! Optional argument: {}{} is not supposed to be a dictionary'.format(upperkeys,key))
#                 else:
#                     check_keys(params[key],d_params[key],upperkeys=upperkeys+'[\'{}\']'.format(key))
                if isdict(d_params[key]):
                    check_keys(params[key],d_params[key],upperkeys=upperkeys+'[\'{}\']'.format(key))
    return True

def isdict(p):
    """Return True if the variable a dictionary."""
    return type(p) is dict

def test_resume(try_resume, params):
    """ Try to load the parameters saved in `params['save_dir']+'params.pkl',`

        Not sure we should implement this function that way.
    """
    resume = False

    if try_resume:
        try:
            with open(params['save_dir']+'params.pkl', 'rb') as f:
                params = pickle.load(f)
            resume = True
            print('Resume, the training will start from the last iteration!')
        except:
            print('No resume, the training will start from the beginning!')

    return resume, params


def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + path)
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + path)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    
    
def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout