class EventBus:
    """Implements a simple event system."""

    subscribers = dict()

    @classmethod
    def unsubscribe(cls, event: str, callback) -> None:
        """Unsubscribes a callback from the event dictionary.

        :param event: Event key
        :type event: str
        :param callback: Event callback function
        :type callback: function
        """

        if not isinstance(event, str):
            raise TypeError("Event key must be a string")
        if not callable(callback):
            raise TypeError(f"Event callback must be callable -> {type(callback)}")
        if not event in cls.subscribers:  # noqa: E713
            return
        if not callback in cls.subscribers[event]:  # noqa: E713
            return
        cls.subscribers[event].remove(callback)

        if len(cls.subscribers[event]) == 0:
            del cls.subscribers[event]

    @classmethod
    def subscribe(cls, event: str, callback) -> None:
        """Subscribes a callback to the event dictionary.

        :param event: Event key
        :type event: str
        :param callback: Event callback function
        :type callback: function
        """
        if not isinstance(event, str):
            raise TypeError("Event key must be a string")
        if not callable(callback):
            raise TypeError(f"Event callback must be callable -> {type(callback)}")
        if not event in cls.subscribers:  # noqa: E713
            cls.subscribers[event] = []
        cls.subscribers[event].append(callback)

    @classmethod
    def post_event(cls, event: str, data):
        """Executes the callbacks when a event occurs.

        :param event: Event key
        :type event: str
        :param data: Data used by the callbacks
        :type data: Any
        """
        if not isinstance(event, str):
            raise TypeError("Event key must be a string")
        if not event in cls.subscribers:  # noqa: E713
            return
        for fn in cls.subscribers[event]:
            try:
                if data is None:
                    fn()
                else:
                    fn(data)
            except Exception as e:
                cls.__logger__.create(str(e.args[0]), "ERR")
