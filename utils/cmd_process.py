import subprocess
import platform


class CmdProcess (object):

    def __init__(self, cmd: str):
        super().__init__()
        self.__cmd = cmd
        self.__result = None
        self.__sys = platform.system()

    def run(self, timeout):
        p = subprocess.Popen(self.__cmd, shell=True)
        try:
            return p.wait(timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            return -1

    def get_result(self):
        return self.__result if self.__sys == "Windows" else (self.__result >> 8)
