import logging
import time
import os


# param:日志级别、日志路径、日志内容、是否打印在控制台
def log_out(log_level, local_business_logs_path, log_content, console_flag):
    rq_ymd = time.strftime(u'%Y-%m-%d', time.localtime(time.time()))
    if not os.path.exists(local_business_logs_path + '/' + rq_ymd):
        os.makedirs(local_business_logs_path + '/' + rq_ymd)
    logger = logging.getLogger()
    formatter = logging.Formatter(u"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if log_level == 'debug':
        logger.setLevel(logging.DEBUG)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'debug.log'
        fh_debug = logging.FileHandler(debug_log_name, mode=u'a')
        fh_debug.setFormatter(formatter)
        logger.addHandler(fh_debug)
        logging.debug(log_content)
        fh_debug.close()
        logger.removeHandler(fh_debug)
    elif log_level == 'info':
        logger.setLevel(logging.INFO)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'info.log'
        fh_info = logging.FileHandler(debug_log_name, mode=u'a')
        fh_info.setFormatter(formatter)
        logger.addHandler(fh_info)
        logging.info(log_content)
        fh_info.close()
        logger.removeHandler(fh_info)
    elif log_level == 'warning':
        logger.setLevel(logging.WARNING)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'warning.log'
        fh_warning = logging.FileHandler(debug_log_name, mode=u'a')
        fh_warning.setFormatter(formatter)
        logger.addHandler(fh_warning)
        logging.warning(log_content)
        fh_warning.close()
        logger.removeHandler(fh_warning)
    elif log_level == 'error':
        logger.setLevel(logging.ERROR)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'error.log'
        fh_error = logging.FileHandler(debug_log_name, mode=u'a')
        fh_error.setFormatter(formatter)
        logger.addHandler(fh_error)
        logging.error(log_content)
        fh_error.close()
        logger.removeHandler(fh_error)
    all_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'all.log'
    fh = logging.FileHandler(all_log_name, mode=u'a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 是否将日志打印在控制台
    if console_flag:
        console = logging.StreamHandler()
        if log_level == 'debug':
            console.setLevel(logging.DEBUG)
        elif log_level == 'info':
            console.setLevel(logging.INFO)
        elif log_level == 'warning':
            console.setLevel(logging.WARNING)
        elif log_level == 'error':
            console.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    logging.debug(log_content)
    logging.info(log_content)
    logging.warning(log_content)
    logging.error(log_content)
    fh.close()
    logger.removeHandler(fh)
