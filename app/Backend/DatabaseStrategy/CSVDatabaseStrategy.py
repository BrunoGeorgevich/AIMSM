from app.Backend.DatabaseStrategy.DatabaseStrategy import DatabaseStrategy

import os


class CSVDatabaseStrategy(DatabaseStrategy):
    def __init__(self) -> None:
        super().__init__()

        self.__database = None

    def open(self):
        if self.__database is not None:
            raise ValueError("Database is already open")

        if os.path.exists("database.csv"):
            self.__database = open("database.csv", "a")
        else:
            self.__database = open("database.csv", "w")
            self.__database.write("FPS;CPU;RAM;GPU;VRAM;Running Models\n")

    def close(self):
        if self.__database:
            self.__database.close()
            self.__database = None
        else:
            raise ValueError("Database is not open")

    def write(self, data):
        rs = data.get("rs", None)
        models = data.get("models", None)

        if rs is None:
            raise ValueError("Pass the 'rs' parameter to the data dict")

        if models is None:
            raise ValueError("Pass the 'models' parameter to the data dict")

        if self.__database:
            self.__database.write(f"{rs.to_csv()};{','.join(models)}\n")
        else:
            raise ValueError("Database is not open")

    def read(self):
        if self.__database:
            self.__database.seek(0)
            return self.__database.read().split("\n")[1:-1]
        else:
            raise ValueError("Database is not open")
