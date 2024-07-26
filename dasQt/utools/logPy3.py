#!/usr/bin/env python
# encoding: utf-8

"""
    * @file: logPy3.py
    * @version: v1.0.0
    * @author: Zhiyu Zhang
    * @desc: 
    * @date: 2024-01-05 20:28:16
    * @Email: erbiaoger@gmail.com
    * @url: erbiaoger.site

"""



import logging
import os
import colorlog
from logging.handlers import RotatingFileHandler
from datetime import datetime


class HandleLog():
    """
    先创建日志记录器（logging.getLogger），然后再设置日志级别（logger.setLevel），
    接着再创建日志文件，也就是日志保存的地方（logging.FileHandler），然后再设置日志格式（logging.Formatter），
    最后再将日志处理程序记录到记录器（addHandler）
    """
 
    def __init__(self, log_name, path="./", level="DEBUG"):
        self.__now_time = datetime.now().strftime('%Y-%m-%d')  # 当前日期格式化
        self.file_basename = log_name 
        log_path = os.path.join(path, 'logs')  # log_path为存放日志的路径
        if not os.path.exists(log_path): os.mkdir(log_path)  # 若不存在logs文件夹，则自动创建

        self.log_colors_config = {
            # 终端输出日志颜色配置
            'DEBUG': 'white',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }

        match level:
            case "DEBUG":
                log_level = logging.DEBUG
            case "INFO":
                log_level = logging.INFO
                print("INFO")
            case "WARNING":
                log_level = logging.WARNING
            case "ERROR":
                log_level = logging.ERROR
            case "CRITICAL":
                log_level = logging.CRITICAL
            case _:
                log_level = logging.INFO
                
        self.log_level = log_level

 
        self.__all_log_path = os.path.join(log_path, log_name + " " + self.__now_time + "-all" + ".log")  # 收集所有日志信息文件
        self.__error_log_path = os.path.join(log_path, log_name + " " + self.__now_time + "-error" + ".log")  # 收集错误日志信息文件
 
        # 配置日志记录器及其级别 设置默认日志记录器记录级别为DEBUG
        self.__logger = logging.getLogger()  # 创建日志记录器
        self.__logger.setLevel(log_level)  # 设置默认日志记录器记录级别
 
    @staticmethod
    def __init_logger_handler(log_path):
        """
        创建日志记录器handler，用于收集日志
        :param log_path: 日志文件路径
        :return: 日志记录器
        """
        # 写入文件，如果文件超过1M大小时，切割日志文件
        logger_handler = RotatingFileHandler(filename=log_path, maxBytes=1 * 1024 * 1024, encoding='utf-8') # 可以设置 backupCount=3 在切割日志文件后仅保留3个文件
        return logger_handler
 
    @staticmethod
    def __init_console_handle():
        """创建终端日志记录器handler，用于输出到控制台"""
        console_handle = colorlog.StreamHandler()
        return console_handle
 
    def __set_log_handler(self, logger_handler, level=logging.DEBUG):
        """
        设置handler级别并添加到logger收集器
        :param logger_handler: 日志记录器
        :param level: 日志记录器级别
        """
        logger_handler.setLevel(level=level)
        self.__logger.addHandler(logger_handler) # 添加到logger收集器
 
    def __set_color_handle(self, console_handle):
        """
        设置handler级别并添加到终端logger收集器
        :param console_handle: 终端日志记录器
        :param level: 日志记录器级别
        """
        console_handle.setLevel(self.log_level)
        self.__logger.addHandler(console_handle)
 
    # @staticmethod
    def __set_color_formatter(self, console_handle, color_config):
        """
        设置输出格式-控制台
        :param console_handle: 终端日志记录器
        :param color_config: 控制台打印颜色配置信息
        :return:
        """
        formatter = colorlog.ColoredFormatter(f'%(log_color)s%(asctime)s %(levelname)s\t{self.file_basename}: %(lineno)d\t%(message)s', log_colors=color_config)
        console_handle.setFormatter(formatter)
 
    # @staticmethod
    def __set_log_formatter(self, file_handler):
        """
        设置日志输出格式-日志文件
        :param file_handler: 日志记录器
        """
        formatter = logging.Formatter(f'%(asctime)s %(levelname)s\t{self.file_basename}: %(lineno)d\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S') # datefmt用于设置asctime的格式，例如：%a, %d %b %Y %H:%M:%S 或者 %Y-%m-%d %H:%M:%S
        file_handler.setFormatter(formatter)
 
    @staticmethod
    def __close_handler(file_handler):
        """
        关闭handler
        :param file_handler: 日志记录器
        """
        file_handler.close()
 
    def __console(self, level, message):
        """构造日志收集器"""
        # 创建日志文件
        all_logger_handler = self.__init_logger_handler(self.__all_log_path) # 收集所有日志文件
        error_logger_handler = self.__init_logger_handler(self.__error_log_path) # 收集错误日志信息文件
        console_handle = self.__init_console_handle()
 
        # 设置日志文件格式
        self.__set_log_formatter(all_logger_handler)
        self.__set_log_formatter(error_logger_handler)
        self.__set_color_formatter(console_handle, self.log_colors_config)
 
        self.__set_log_handler(all_logger_handler, level=self.log_level)  # 设置handler级别并添加到logger收集器
        self.__set_log_handler(error_logger_handler, level=logging.ERROR)
        self.__set_color_handle(console_handle)
 
            
        match level:
            case "info":
                self.__logger.info(message)
            case "debug":
                self.__logger.debug(message)
            case "warning":
                self.__logger.warning(message)
            case "error":
                self.__logger.exception(message)
            case "critical":
                self.__logger.exception(message)
            case _:
                self.__logger.info(message)
            
        
        self.__logger.removeHandler(all_logger_handler)  # 避免日志输出重复问题
        self.__logger.removeHandler(error_logger_handler)
        self.__logger.removeHandler(console_handle)
 
        self.__close_handler(all_logger_handler)  # 关闭handler
        self.__close_handler(error_logger_handler)
 
    def debug(self, message):
        self.__console('debug', message)
 
    def info(self, message):
        self.__console('info', message)
 
    def warning(self, message):
        self.__console('warning', message)
 
    def error(self, message):
        self.__console('error', message)
 
    def critical(self, message):
        self.__console('critical', message)
 
if __name__ == '__main__':
    log = HandleLog(os.path.split(__file__)[-1].split(".")[0])
    log.info("这是日志信息")
    log.debug("这是debug信息")
    log.warning("这是警告信息")
    log.error("这是错误日志信息")
    log.critical("这是严重级别信息")