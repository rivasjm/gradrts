import time


class ExecTime:
    def __init__(self):
        self.exec_time = -1
        self._before = -1

    def init(self):
        self.exec_time = -1
        self._before = time.perf_counter()

    def stop(self):
        if self._before > -1:
            self.exec_time = time.perf_counter() - self._before

    def has_time(self):
        return self.exec_time > -1
