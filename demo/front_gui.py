import re
import streamlit as st

from async_request import main


st.title('Chatting')
st.text('\n')
st.text('\n')
st.markdown('<b>Instruction</b>', unsafe_allow_html=True)
instruction = st.text_area('', key='instruction')

instruction = re.sub(r'\n+', '\n', instruction)
instructions = instruction.split('\n')
instruction = instructions[0].strip()
description = '\n'.join(instructions[1:]).strip()

if instruction:
    with st.spinner("processing..."):
        url = ['http://127.0.0.1:8502']

        answers = main(url, instruction, description) 
        for ans in answers:
            if 'response' in ans.keys():
                response = ans['response']
        

        st.markdown('#### Response')
        st.write('{}'.format(response))