import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle

def arg_helper(params, d_params):
    """Check if all parameter of d_params are in params. If not, they are added to params."""
    for key in d_params.keys():
        params[key] = params.get(key, d_params[key])
        if type(params[key]) is dict:
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
                if not(isdict(d_params[key])):
                    print('Warning! Optional argument: {}{} is not supposed to be a dictionary'.format(upperkeys,key))
                else:
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