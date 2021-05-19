import os
import server


class Application:

    def __init__(self, frontend_port=8080, backend_port=8081):
        os.system(
            f"streamlit run {os.path.abspath(server.__file__)} --server.port={frontend_port}"
        )
