#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(
    page_title="Prompt Engineering Playground",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the text generation pipeline
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

text_generator = load_model()


# Custom CSS for vibrant UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        color: #000000;
    }
    .stApp {
        background: transparent;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    .css-1fcdlhh {
        background-color: #ffffff !important;
    }
    .css-1fcdlhh:hover {
        background-color: #f2f2f2 !important;
    }
    .stButton > button {
        background-color: #ff6f61;
        color: white;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        height: 50px;
        width: 200px;
    }
    .stButton > button:hover {
        background-color: #ff856e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("✨ Prompt Engineering Playground with GPT-2 ✨")

# Sidebar: Prompt settings
st.sidebar.header("🌟 Settings")
st.sidebar.markdown(
    """
    Customize your text generation settings here:
    """
)
max_length = st.sidebar.slider("Max Length:", min_value=10, max_value=200, value=50)
temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.5, step=0.1, value=0.7)
top_k = st.sidebar.slider("Top-k Sampling:", min_value=1, max_value=50, value=10)

# Main: User prompt input
st.header("🎯 Experiment with Prompts")
prompt = st.text_area("Enter your prompt:", "Once upon a time in a world powered by AI,")
st.markdown("---")

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button("🚀 Generate Text")

if generate_btn:
    with st.spinner("Generating response..."):
        try:
            response = text_generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=1,
            )
            result = response[0]["generated_text"]
            st.success("Generated Text:")
            st.write(
                f"""
                <div style="background: #ffffff; color: #000000; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); font-size: 16px;">
                {result}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error: {e}")

# Prompt engineering tips
st.markdown("---")
st.subheader("💡 Prompt Engineering Tips:")
st.markdown(
    """
    - **Be Specific**: Provide detailed and explicit instructions.
    - **Set the Context**: Use contextual clues to guide the model's response.
    - **Experiment with Parameters**: Adjust `temperature`, `max_length`, and `top_k` for diverse outputs.
    """
)


# In[ ]:




