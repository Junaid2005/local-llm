import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        dtype="auto",
    )
    return tokenizer, model


def main():
    st.set_page_config(page_title="Qwen Chat", page_icon="💬")
    st.title("Qwen Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    tokenizer, model = load_model()

    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            assistant_reply = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            st.markdown(assistant_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )


if __name__ == "__main__":
    main()
