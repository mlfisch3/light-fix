import streamlit as st
import gc
import sys
import datetime

VERSION = 2
title = f'SODEF{VERSION}'
st.set_page_config(page_title=title, layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

from app import run_app
from utils.logging import timestamp, log_memory

if __name__ == '__main__':

    total_start = datetime.datetime.now()
    log_memory('main|run_app|B')


    run_app()

   # gc.collect()
    log_memory('main|run_app|E')
    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()
    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')
    sys.stdout.flush()