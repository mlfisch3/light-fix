import streamlit as st
import datetime
from app import run_app
from utils.logging import timestamp

if __name__ == '__main__':

    total_start = datetime.datetime.now()

    run_app()

    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()
    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')
