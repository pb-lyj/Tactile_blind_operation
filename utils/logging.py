"""
日志工具 - 用于同时输出到控制台和文件
"""

import sys

class Logger:
    """同时写入控制台和文件的日志类"""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
