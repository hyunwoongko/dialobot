from flask import Flask


class Backend:

    def __init__(self, port):
        self.app = Flask(__name__)
        self.build()
        self.app.run(port=port, host="0.0.0.0")

    def build(self):

        @self.app.route('/')
        def index() -> str:
            return "Dialobot Backend Server"

        @self.app.route('/backend')
        def backend() -> str:
            return "BACKEND"
