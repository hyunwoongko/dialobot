import os
from dialobot.app import server


class Application:

    def __init__(self, frontend_port=8080, backend_port=8081):

        os.system(
            f"streamlit run --server.port={frontend_port} {os.path.abspath(server.__file__)} -- --backend_port={backend_port}"
        )
