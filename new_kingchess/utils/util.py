import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO,  # 设置日志级别为DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
                    datefmt='%Y-%m-%d %H:%M:%S',  # 日期格式
                    filename='kingchess.log',  # 日志输出文件
                    filemode='w')  # 文件模式，'w'为覆盖写入

# 创建日志记录器
LOGGER = logging.getLogger('kingchess')

# 记录不同级别的日志
LOGGER.debug('This is a debug message')
LOGGER.info('This is an info message')
LOGGER.warning('This is a warning message')
LOGGER.error('This is an error message')
LOGGER.critical('This is a critical message')


if __name__ == '__main__':
    LOGGER.info("hello")