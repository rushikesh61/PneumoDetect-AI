import streamlit as st

st.set_page_config(page_title="HTML Test")

st.title("PneumoDetect AI Test")

st.markdown(
    """
    <div style="background:#073b4c;padding:30px;border-radius:20px;color:white;">
        <h1>PneumoDetect AI</h1>
        <p>HTML rendering is working.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.success("If the box above is styled, HTML is working correctly.")
