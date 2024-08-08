from time import time

class Logger:

    def __init__(self, path: str):
        self.path = path

    def log(self, message: str):
        with open(self.path, "a") as f:
           f.write(message)
           f.flush()

class Profiler:

    def __init__(self, description: str):
        self.logger = Logger("/export/data/isulzer/DON-vs-NODE/profiling.log")
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()
        # print(f"\n{self.description}: {self.end_time - self.start_time}")
        self.logger.log(f"\n{self.description}: {self.end_time - self.start_time}")