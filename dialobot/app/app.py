from dialobot.app.backend import Backend
from dialobot.app.frontend import Frontend


class Application:

    def __init__(self):
        self.frontend = Frontend()
        self.backend = Backend()
