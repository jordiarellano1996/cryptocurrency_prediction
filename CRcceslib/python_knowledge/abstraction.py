import abc


class PluginBase(abc.ABC):

    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source
        and return an object.
        """
        print(input)

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""

    @abc.abstractproperty
    def color(self):
        return "negro"


class SubclassImplementation(PluginBase):

    def load(self, _input):
        super().load("Funcionalida abstraccion")
        return _input

    def save(self, output, data):
        return output, data

    @property
    def color(self):
        return "rojo"


if __name__ == '__main__':
    try:
        f = PluginBase()
    except TypeError:
        print("Error: Can't instantiate abstract class PluginBase with abstract methods color, load, save")

    x = SubclassImplementation()
    print(x.load("ff"))
    print(x.color)
