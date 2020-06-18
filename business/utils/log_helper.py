import logging
import time
import os


def log_out(log_level, local_business_logs_path, log_content):
    rq = time.strftime(u'%Y%m%d', time.localtime(time.time()))
    if not os.path.exists(local_business_logs_path):
        os.makedirs(local_business_logs_path)
    log_path = local_business_logs_path
    log_name = log_path + '/' + rq + u'.logs'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_name, mode=u'a')
    formatter = logging.Formatter(u"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if log_level == 'debug':
        logging.debug(log_content)
    elif log_level == 'info':
        logging.info(log_content)
    elif log_level == 'warning':
        logging.warning(log_content)
    elif log_level == 'error':
        logging.error(log_content)
    elif log_level == 'critical':
        logging.critical(log_content)
    else:
        logging.error("日志级别错误！")
    fh.close()
    logger.removeHandler(fh)


def logs_range_clear():
    pass


def try_except(print_debug=True):
    def inner1(f):
        def inner2(*args, **kwargs):
            try:
                res = f(*args, **kwargs)
            except Exception as err:
                if print_debug:
                    import sys  # 这里导入模块是为了方便直接复制使用，现实中这个应该放到文件头部
                    info = sys.exc_info()[2].tb_frame.f_back
                    temp = "filename:{}\nlines:{}\tfuncation:{}\terror:{}"
                    log_out('error', temp.format(info.f_code.co_filename, info.f_lineno, f.__name__, repr(err)),'')
                res = 'Error occurred ' + info.f_code.co_filename
            return res
        return inner2
    return inner1

# @try_except()
# def div(a, b):
#     a= float(a) / float(b)

# ret = div(3, 'mo')
# print(ret)