import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import streamlit as st

system_prompt = """
You are a science communicator who is experienced in liasing with government officials. 
Given a plot containing information about notifiable diseases, your job is to answer any questions a UK minister may have about what the plot is showing. 
Make your response easy to understand to someone with no technical knowledge.

DO: 
* do make your responses easily understandable to someone without any knowledge of STEM subjects.
* do keep your responses concise - try to limit your responses to 100 words each. 


DO NOT:
* do not make up any information. use only information present in the plot and given by the user. 
* do not use technical language unless specifically asked about. for example, use \"spread\" instead of \"standard deviation\". 
* do not give recommendations for policy interventions or what should be done using the information in the plot. Do not make predictions about the future. If asked about this, you can respond with something like \"I\'m sorry, but I can\'t help with that. Policy design requires additional information that I do not have access to, and it is best to consult a colleague or the team that produced this briefing.\"
"""

generation_config = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

vertexai.init(project="anc-pg-sbox-team-20", location="us-central1")
model = GenerativeModel(
    "gemini-1.5-pro-preview-0409",
    system_instruction=[system_prompt]
)

####### functions

def get_image(fp):
    with open(fp, 'rb') as f:
        image = f.read()

    encoded_image = Part.from_data(
        mime_type="image/jpeg",
        data=base64.b64encode(image).decode('utf-8'))
    
    return image, encoded_image

def get_response(input):

    response = st.session_state.model.send_message(
        [st.session_state.encoded_image, input],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return response

def get_first_response():
    response = st.session_state.model.send_message(
        [st.session_state.encoded_image, 'What can you tell me about this plot?'],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return response

####### streamlit
st.session_state.update(st.session_state)
st.session_state.model = model.start_chat()

with st.sidebar():
    with st.container(border=True):
        # initialise chat history
        if "messages" not in st.session_state:
            st.session_state.is_first = True
            st.session_state.image, st.session_state.encoded_image = get_image('covid.jpeg')
            st.session_state.first_message = f"What can you tell me about this plot? {st.session_state.encoded_image}"

            st.session_state["messages"] = [
                {
                    'role': 'assistant',
                    'content': "Hello! I'm here to help you with any questions regarding the plots!"
                },
                {
                    'role': 'user',
                    'content': 'What can you tell me about this plot?',
                },

            ]

        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message['content'] == 'What can you tell me about this plot?':
                    st.image(st.session_state.image)
                    st.markdown('What can you tell me about this plot?')
                else:
                    st.markdown(message["content"])


        # new user input
        if question := st.chat_input('Ask me anything...'):
            st.session_state.messages.append(
                {
                    'role': 'user',
                    'content': question
                }
            )
            with st.chat_message('user'):
                st.markdown(question)

        if st.session_state.messages[-1]['role'] != 'assistant':
            with st.chat_message('assistant'):
                with st.spinner('Thinking...'):
                    if st.session_state.is_first:
                        response = get_first_response().text
                        st.session_state.is_first = False
                    else:
                        response = get_response(st.session_state.messages[-1]['content']).text
                    
                    st.markdown(response)
            message = {'role': 'assistant', 'content': response}
            st.session_state.messages.append(message)



