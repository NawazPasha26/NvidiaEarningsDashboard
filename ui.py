"""
@author - Nawaz Pasha
Updated on Thu Nov 20 17:16:47 2025 
UI related style updates
"""

#ui.py
#This module has the UI styles defined along with some basic branding
import os
import streamlit as st
import pandas as pd
from typing import Optional

def use_global_css():
    st.markdown(
        """
        <style>
        /* Sidebar tidy */
        section[data-testid="stSidebar"] pre,
        section[data-testid="stSidebar"] code { display: none !important; }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .5rem; }

        /* Tabs: wrap */
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap !important; row-gap: 4px !important; }
        .stTabs [data-baseweb="tab"] { height:auto !important; white-space:normal !important; line-height:1.2 !important;
                                      padding:6px !important; margin-right:6px !important; border-radius:6px !important; }
        .stTabs [data-baseweb="tab"] p { margin:0 !important; }
        .stTabs [role="tablist"]::-webkit-scrollbar { display:none !important; }

        .modebar { z-index:10; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def control_panel_branding():
    with st.sidebar:
        with st.container():
            logo_path = "nvidia_logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path)
            else:
                st.markdown("### **NVIDIA**")
        st.markdown("---")
        st.markdown("**Developer - Nawaz Pasha**")
        st.markdown("📞 +91-9986551895")
        st.markdown("✉️ Navvu18@gmail.com")
        st.markdown("---")


def df_show(df: pd.DataFrame, fmt: Optional[dict] = None):
    # Use 'stretch' to mimic previous use_container_width behavior
    st.dataframe(df.style.format(fmt) if fmt else df, use_container_width=True)



