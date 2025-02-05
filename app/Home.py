import streamlit as st

from phi.tools.streamlit.components import check_password

st.set_page_config(
    page_title="Parkstreet AI!",
    page_icon=":rocket:",
)
st.title(":rocket: Parkstreet AI!")


def main() -> None:
    st.markdown("---")
    st.markdown("## Select an App from the sidebar:")
    st.markdown("#### 1. Parkstreet AI")
    st.markdown("#### 2. Parkstreet Example AI")

    st.sidebar.success("Select App from above")


if check_password():
    main()
