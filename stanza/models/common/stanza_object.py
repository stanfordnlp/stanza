def _readonly_setter(self, name):
    full_classname = self.__class__.__module__
    if full_classname is None:
        full_classname = self.__class__.__qualname__
    else:
        full_classname += '.' + self.__class__.__qualname__
    raise ValueError(f'Property "{name}" of "{full_classname}" is read-only.')

class StanzaObject(object):
    """
    Base class for all Stanza data objects that allows for some flexibility handling annotations
    """

    @classmethod
    def add_property(cls, name, default=None, getter=None, setter=None):
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

