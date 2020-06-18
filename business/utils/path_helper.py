import os


def get_local_project_path(path, level):
    ret_path = path
    if level == 0:
        return ret_path
    return get_local_project_path(
        os.path.abspath(os.path.join(ret_path, os.path.pardir)), level - 1)


def trans_symbol_to_word(symbol):
    word = symbol
    return word.replace('|~|', '/')


def delete_file_in_path(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            delete_file_in_path(c_path)
        else:
            os.remove(c_path)
