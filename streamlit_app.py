import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import time
import numpy as np
import pygame
import streamlit as st
import app as original_app  

st.set_page_config(page_title="Kado Surprise", page_icon="🎁", layout="centered")

if "initialized" not in st.session_state:
    pygame.init()
    st.session_state.screen = pygame.Surface((original_app.W, original_app.H))
    st.session_state.font = pygame.font.Font(None, 28)
    st.session_state.show = original_app.Surprise()
    st.session_state.initialized = True

col1, col2 = st.columns(2)
with col1:
    if st.button("🎁 Buka Kado", use_container_width=True, type="primary"):
        st.session_state.show.open()
with col2:
    if st.button("🔄 Reset (R)", use_container_width=True):
        st.session_state.show.reset()

frame_placeholder = st.empty()

dt = 1.0 / original_app.FPS
while True:
    st.session_state.show.update(dt)
    st.session_state.show.draw(st.session_state.screen, st.session_state.font)
    
    frame = np.transpose(pygame.surfarray.array3d(st.session_state.screen), (1, 0, 2))
    frame_placeholder.image(frame, use_container_width=True)
    time.sleep(dt)