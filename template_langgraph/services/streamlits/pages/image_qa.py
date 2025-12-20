import base64
from os import getenv

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

DEFAULT_SYSTEM_PROMPT = getenv(
    "IMAGE_QA_SYSTEM_PROMPT",
    "You are a professional image analyst. Answer questions about the image provided.",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "encoded_image" not in st.session_state:
    st.session_state.encoded_image = None
if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False
if "captured_image_bytes" not in st.session_state:
    st.session_state.captured_image_bytes = None

with st.sidebar:
    azure_openai_endpoint = st.text_input(
        label="AZURE_OPENAI_ENDPOINT",
        value=getenv("AZURE_OPENAI_ENDPOINT"),
        key="AZURE_OPENAI_ENDPOINT",
        type="default",
    )
    azure_openai_api_key = st.text_input(
        label="AZURE_OPENAI_API_KEY",
        key="AZURE_OPENAI_API_KEY",
        type="password",
    )
    azure_openai_api_version = st.text_input(
        label="AZURE_OPENAI_API_VERSION",
        value=getenv("AZURE_OPENAI_API_VERSION"),
        key="AZURE_OPENAI_API_VERSION",
        type="default",
    )
    azure_openai_gpt_model = st.text_input(
        label="AZURE_OPENAI_GPT_MODEL",
        value=getenv("AZURE_OPENAI_GPT_MODEL"),
        key="AZURE_OPENAI_GPT_MODEL",
        type="default",
    )

    st.divider()
    system_prompt = st.text_area(
        label="System Prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        key="system_prompt",
        height=150,
    )

    "[Azure Portal](https://portal.azure.com/)"

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.encoded_image = None
        st.session_state.image_analyzed = False
        st.session_state.captured_image_bytes = None
        st.rerun()


def is_configured():
    return azure_openai_api_key and azure_openai_endpoint and azure_openai_api_version and azure_openai_gpt_model


def get_client():
    return AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
    )


def chat_with_image(client, messages, encoded_image, system_prompt_text, include_image=True):
    """Send chat request with optional image context."""
    api_messages = [
        {
            "role": "system",
            "content": system_prompt_text,
        }
    ]

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            # Include image only for the first user message or when explicitly needed
            if include_image and i == 0:
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                            {
                                "type": "text",
                                "text": msg["content"],
                            },
                        ],
                    }
                )
            else:
                api_messages.append({"role": "user", "content": msg["content"]})
        else:
            api_messages.append({"role": "assistant", "content": msg["content"]})

    response = client.chat.completions.create(
        model=azure_openai_gpt_model,
        messages=api_messages,
    )
    return response.choices[0].message.content


st.title("📷 Camera Image Q&A")

if not is_configured():
    st.warning("Please fill in the required fields at the sidebar.")

st.info("Take a photo with your camera. AI will describe the image and you can ask follow-up questions.")

# Camera input
camera_image = st.camera_input("Take a picture")

if camera_image:
    # Read and store the image bytes
    image_bytes = camera_image.getvalue()
    new_encoded_image = base64.b64encode(image_bytes).decode()

    # Check if this is a new image
    if st.session_state.encoded_image != new_encoded_image:
        st.session_state.encoded_image = new_encoded_image
        st.session_state.captured_image_bytes = image_bytes
        st.session_state.messages = []
        st.session_state.image_analyzed = False

# Display captured image and chat if we have an image
if st.session_state.encoded_image and st.session_state.captured_image_bytes:
    # Show the captured image
    st.image(st.session_state.captured_image_bytes, caption="Captured Image", use_container_width=True)

    # Initial image analysis
    if not st.session_state.image_analyzed and is_configured():
        with st.spinner("Analyzing image..."):
            client = get_client()
            initial_question = "Please describe the content of this image in detail."
            st.session_state.messages.append({"role": "user", "content": initial_question})

            response = chat_with_image(
                client,
                st.session_state.messages,
                st.session_state.encoded_image,
                system_prompt,
                include_image=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.image_analyzed = True

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for follow-up questions
    if prompt := st.chat_input("Ask a question about the image..."):
        if is_configured():
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = get_client()
                    response = chat_with_image(
                        client,
                        st.session_state.messages,
                        st.session_state.encoded_image,
                        system_prompt,
                        include_image=True,
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please configure Azure OpenAI settings in the sidebar.")
else:
    st.info("📸 Please take a photo to start the conversation.")
