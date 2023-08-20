class Struct:
    def __init__(self, data=None, **kwargs):
        if data is None:
            data = kwargs
        for name, value in data.items():
            my_value = self._wrap(value)
            setattr(self, name, my_value)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        my_str = "Struct("

        values = []
        for key, value in self.__dict__.items():
            if key == "_id":
                continue
            values.append(f"{key}={value}")
        my_str += ", ".join(values)

        my_str += ")"
        return my_str
