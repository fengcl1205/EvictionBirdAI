import logging
import time
import os
from logging.handlers import RotatingFileHandler
from business.utils import path_helper as ph
from business.utils import yaml_helper
project_address = ph.get_local_project_path(os.path.dirname(os.path.abspath(__file__)), 1)
business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
local_business_logs_path = business_path_config['local_business_logs_path']
print_console_flag = business_path_config['print_console_flag']
log_file_backup_count = business_path_config['log_file_backup_count']
log_file_limit_size = business_path_config['log_file_limit_size']


# param:日志级别、日志路径、日志内容、是否打印在控制台
def log_out(log_level, log_content):
    rq_ymd = time.strftime(u'%Y-%m-%d', time.localtime(time.time()))
    if not os.path.exists(local_business_logs_path + '/' + rq_ymd):
        os.makedirs(local_business_logs_path + '/' + rq_ymd)
    logger = logging.getLogger()
    formatter = logging.Formatter(u"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if log_level == 'debug':
        logger.setLevel(logging.DEBUG)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'debug.log'
        # maxBytes：最大10M， backupCount：备份数量
        fh_debug = RotatingFileHandler(debug_log_name, mode=u'a', maxBytes=log_file_limit_size * 1024 * 1024,
                                      backupCount=log_file_backup_count, encoding=None, delay=0)
        fh_debug.setFormatter(formatter)
        logger.addHandler(fh_debug)
        logging.debug(log_content)
        fh_debug.close()
        logger.removeHandler(fh_debug)
    elif log_level == 'info':
        logger.setLevel(logging.INFO)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'info.log'
        fh_info = RotatingFileHandler(debug_log_name, mode=u'a', maxBytes=log_file_limit_size * 1024 * 1024,
                                      backupCount=log_file_backup_count, encoding=None, delay=0)
        fh_info.setFormatter(formatter)
        logger.addHandler(fh_info)
        logging.info(log_content)
        fh_info.close()
        logger.removeHandler(fh_info)
    elif log_level == 'warning':
        logger.setLevel(logging.WARNING)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'warning.log'
        fh_warning = RotatingFileHandler(debug_log_name, mode=u'a', maxBytes=log_file_limit_size * 1024 * 1024,
                                      backupCount=log_file_backup_count, encoding=None, delay=0)
        fh_warning.setFormatter(formatter)
        logger.addHandler(fh_warning)
        logging.warning(log_content)
        fh_warning.close()
        logger.removeHandler(fh_warning)
    elif log_level == 'error':
        logger.setLevel(logging.ERROR)
        debug_log_name = local_business_logs_path + '/' + rq_ymd + '/' + u'error.log'
        fh_error = RotatingFileHandler(debug_log_name, mode=u'a', maxBytes=log_file_limit_size * 1024 * 1024,
                                      backupCount=log_file_backup_count, encoding=None, delay=0)
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
    if print_console_flag:
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
