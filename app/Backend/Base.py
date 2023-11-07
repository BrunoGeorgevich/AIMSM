import numpy as np
from PySide2.QtCore import (
    Property,
    QAbstractListModel,
    QModelIndex,
    Qt,
    Signal,
    Slot,
)
from PySide2.QtQuick import QQuickImageProvider


class Model(QAbstractListModel):
    """Class responsible to handle data used to
    generate tables in the frontend.
    """

    __entries = []
    __roles_names = dict()
    __roles_actions = dict()
    __page_data = []
    __filtered_data = []
    __current_page = 0
    __page_size = 8

    def __init__(self, roles, parent=None):
        super(Model, self).__init__(parent)

        initial_index = Qt.UserRole + 1000
        self.__roles_names = dict()
        self.__roles_actions = dict()

        for role in roles:
            self.__roles_names[initial_index] = role
            self.__roles_actions[initial_index] = roles[role]
            initial_index += 1

    def add_role(self, role, action):  # TODO finalizar documentação
        current_index = Qt.UserRole + 1000 + len(self.__roles_names) + 1
        self.__roles_names[current_index] = role
        self.__roles_actions[current_index] = action

    filtered_data_is_empty_changed = Signal()

    @Property("QVariant", notify=filtered_data_is_empty_changed)
    def filteredDataIsEmpty(self) -> bool:
        """Method that evaluates if the current
        filtered data is empty.

        :return: Whether current filtered data is empty.
        :rtype: bool
        """
        return len(self.__filtered_data) <= 0

    @Slot(result=int)
    def realCount(self, parent=QModelIndex()) -> int:  # TODO concluir documentação
        """Method that retrieves the current amount of entries.

        :param parent: _description_, defaults to QModelIndex()
        :type parent: _type_, optional
        :return: Current number of entries.
        :rtype: int
        """
        if parent.isValid():
            return 0
        return len(self.__entries)

    @Slot(result=int)
    def rowCount(self, parent=QModelIndex()) -> int:  # TODO concluir documetação
        """Returns current number of table rows.

        :param parent: _description_, defaults to QModelIndex()
        :type parent: _type_, optional
        :return: Current number of table rows.
        :rtype: int
        """
        if parent.isValid():
            return 0
        return (
            self.__page_size
            if len(self.__filtered_data) > self.__page_size
            else len(self.__filtered_data)
        )

    def appendRow(self, obj):  # TODO concluir documentação
        """Method that appends a row to the table.

        :param obj: _description_
        :type obj: _type_
        """
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.__entries.append(obj)
        self.endInsertRows()

    def removeRow(self, index):  # TODO concluir documentação
        """Method that removes a row from the table.

        :param index: _description_
        :type index: _type_
        """
        self.beginRemoveRows(QModelIndex(), index, index)
        del self.__entries[index]
        self.endRemoveRows()

    def firstRow(self):  # TODO concluir documentação
        """Method that retrieves the first row from the table.

        :return: _description_
        :rtype: _type_
        """
        if len(self.__entries) > 0:
            return self.__entries[0]

    def lastRow(self):  # TODO concluir documentação
        """Method that retrieves the last row from the table.

        :return: _description_
        :rtype: _type_
        """
        if len(self.__entries) > 0:
            return self.__entries[-1]

    isEmptyChanged = Signal()

    @Property(bool, notify=isEmptyChanged)
    def isEmpty(self) -> bool:
        """Method that evaluates whether current entries are empty.

        :return: Whether current entries are empty or not.
        :rtype: bool
        """
        return len(self.__entries) == 0

    def clear(self):
        """Method that clears current table entries."""
        self.beginRemoveRows(QModelIndex(), 0, self.rowCount())
        self.__entries = []
        self.endRemoveRows()

    def data(self, index, role=Qt.DisplayRole):  # TODO concluir a documentação
        if 0 <= index.row() < self.rowCount() and index.isValid():
            item = self.page_data[index.row()]
            return self.__roles_actions[role](item)
        else:
            return None

    def roleNames(self) -> dict:
        """Getter for current role names.

        :return: Role names.
        :rtype: dict
        """
        return self.__roles_names

    def set_entries(self, entries: list):
        """Setter for table entries.

        :param entries: Desired entries.
        :type entries: list
        """
        self.__entries = entries
        self.__filtered_data = entries

    @Slot(int)
    def changePage(self, page_num):  # TODO concluir documentação
        if page_num >= 0:
            self.__current_page = page_num

        if self.__current_page > self.num_pages - 1:
            self.__current_page = self.num_pages - 1

        if self.__current_page < 0:
            self.__current_page = 0

        if (self.__current_page + 1) * self.__page_size > len(self.__filtered_data):
            page_data = self.__filtered_data[self.__page_size * self.__current_page :]
            num_of_itens = len(page_data)
        else:
            page_data = self.__filtered_data[
                self.__page_size
                * self.__current_page : self.__page_size
                * (self.__current_page + 1)
            ]
            num_of_itens = self.__page_size

        self.__page_data = page_data

        self.beginRemoveRows(QModelIndex(), 0, self.__page_size)
        self.endRemoveRows()
        if num_of_itens > 0:
            self.beginInsertRows(QModelIndex(), 0, num_of_itens - 1)
            self.endInsertRows()

        self.current_page_changed.emit()
        self.num_pages_changed.emit()
        self.isEmptyChanged.emit()

    def filter_data(self, data):  # TODO concluir documentação
        """Method that retrieves data resulting from
        an input filter.

        :param data: _description_
        :type data: _type_
        """
        self.__filtered_data = data
        self.changePage(-1)
        self.num_pages_changed.emit()

    current_page_changed = Signal()

    @Property(int, notify=current_page_changed)
    def current_page(self):  # TODO concluir documentação
        return self.__current_page + 1

    num_pages_changed = Signal()

    @Property(int, notify=num_pages_changed)  # TODO concluir documentação
    def num_pages(self):
        return (len(self.__filtered_data) // self.__page_size) + (
            1 if len(self.__filtered_data) % self.__page_size > 0 else 0
        )

    entries_changed = Signal()

    @Property(np.ndarray, fset=set_entries, notify=entries_changed)
    def entries(self):
        return self.__entries  # TODO concluir documentação

    pageDataChanged = Signal()

    @Property(np.ndarray, notify=pageDataChanged)
    def page_data(self):
        return self.__page_data  # TODO concluir documentação


class ImageProvider(QQuickImageProvider):
    """Class responsible for delivering image data from the
    backend to the frontend.
    """

    __image_provider_handler = None

    def __init__(self, image_provider_handler):
        super().__init__(QQuickImageProvider.Image)
        self.__image_provider_handler = image_provider_handler

    def requestImage(self, path: str, size, requestedSize):
        """Retrieves image from the backend to the frontend.

        :param path: String containing data necessary to the request.
        :type path: str
        :param size: _description_
        :type size: _type_
        :param requestedSize: _description_
        :type requestedSize: _type_
        :return: Requested image.
        :rtype: QImage
        """
        if self.__image_provider_handler is not None:
            return self.__image_provider_handler(path, size, requestedSize)
        else:
            print("No image provider handler defiend!")
