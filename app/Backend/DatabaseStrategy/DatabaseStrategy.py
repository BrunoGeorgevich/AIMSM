from typing import Protocol


class DatabaseStrategy(Protocol):
    def open(self):
        ...

    def close(self):
        ...

    def write(self, data):
        ...

    def read(self):
        ...
