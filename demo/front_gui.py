import streamlit as st

from async_request import main


st.title('Chatting')
st.text('\n')
st.text('\n')
st.markdown('<b>Instruction</b>', unsafe_allow_html=True)
instruction = st.text_input('', key='instruction')
st.markdown('<b>Description</b>', unsafe_allow_html=True)
description = st.text_input('', key='description')
st.text('\n')
st.text('\n')


if st.button('Send'):
    with st.spinner("processing..."):
        url = ['http://127.0.0.1:8502']

        answers = main(url, instruction, description) 
        for ans in answers:
            if 'response' in ans.keys():
                response = ans['response']
        

        st.markdown('#### Response')
        st.write('{}'.format(response))