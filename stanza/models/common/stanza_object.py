from typing import Callable, NoReturn, Optional, Type, TypeVar


_StanzaObjectT = TypeVar("_StanzaObjectT", bound="StanzaObject")
# A property's backing value, getter result, and setter input can intentionally differ.
_PropertyDefaultT = TypeVar("_PropertyDefaultT")
_PropertyValueT = TypeVar("_PropertyValueT")
_PropertyInputT = TypeVar("_PropertyInputT")


def _qualified_class_name(module: Optional[str], qualname: str) -> str:
    return qualname if module is None else f"{module}.{qualname}"


def _readonly_setter(instance: "StanzaObject", name: str) -> NoReturn:
    full_classname = _qualified_class_name(
        instance.__class__.__module__,
        instance.__class__.__qualname__,
    )
    raise ValueError(f'Property "{name}" of "{full_classname}" is read-only.')


class StanzaObject:
    """
    Base class for all Stanza data objects that allows for some flexibility handling annotations
    """

    @classmethod
    def add_property(
        cls: Type[_StanzaObjectT],
        name: str,
        default: Optional[_PropertyDefaultT] = None,
        getter: Optional[Callable[[_StanzaObjectT], _PropertyValueT]] = None,
        setter: Optional[Callable[[_StanzaObjectT, _PropertyInputT], None]] = None,
    ) -> None:
        """
        Add a property accessible through self.{name} with underlying variable self._{name}.
        Optionally setup a setter as well.
        """

        if hasattr(cls, name):
            raise ValueError(f'Property by the name of {name} already exists in {cls}. Maybe you want to find another name?')

        setattr(cls, f'_{name}', default)
        if getter is None:
            getter = lambda self: getattr(self, f'_{name}')
        if setter is None:
            setter = lambda self, value: _readonly_setter(self, name)

        setattr(cls, name, property(getter, setter))
