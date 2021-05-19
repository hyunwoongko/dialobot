if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--backend_port", type=str, default=None)
    args = parser.parse_args()

    import streamlit as st
    from dialobot.app.frontend.pages import loading
    if not hasattr(st, "already_started_server"):
        st.already_started_server = True
        loading.page()

        # thread hijacking
        from dialobot.app.backend import Backend
        Backend(args.backend_port)

    from dialobot.app.frontend import Frontend
    Frontend()
