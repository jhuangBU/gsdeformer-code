import asyncio
from typing import TypeVar, Generic, Callable, List, Coroutine

T = TypeVar("T")


class EventEmitter(Generic[T]):

    def __init__(self):
        self._callbacks = []

    def register(self, function: Callable[[T], None]) -> Callable[[], None]:
        self._callbacks.append(function)
        return lambda: self._callbacks.remove(function)

    def emit(self, val: T):
        for f in self._callbacks:
            f(val)


class ValueEmitter(EventEmitter[T]):

    def __init__(self, init_val: T):
        super().__init__()
        self._val = init_val

    @property
    def val(self) -> T:
        return self._val

    def emit(self, val: T):
        self._val = val
        for f in self._callbacks:
            f(val)

    def notify_inplace_mutation(self):
        self.emit(self.val)