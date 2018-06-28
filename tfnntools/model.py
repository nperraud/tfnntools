def rprint(msg, reuse=False):
    if not reuse:
        print(msg)

class BaseNet(object):
    """Base Net abstract class."""

    def __init__(self, params, name="BaseNet"):
        self._params = params
        self._name = name
        self._build_net()

    def _build_net(self, inputs):
        raise ValueError('This is an abstract class')


    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def outputs(self):
        return self._outputs