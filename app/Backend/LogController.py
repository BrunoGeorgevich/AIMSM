from app.Backend.DatabaseStrategy.DatabaseStrategy import DatabaseStrategy


class LogController:
    def __init__(self, database_strategy: DatabaseStrategy) -> None:
        self.__database = database_strategy()

    def open_database(self):
        self.__database.open()

    def close_database(self):
        self.__database.close()

    def write_to_database(self, data):
        self.__database.write(data)

    def read_from_database(self):
        return self.__database.read()
