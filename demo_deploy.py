import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import os


#%% Load features dataset
path_data = 'https://raw.githubusercontent.com/PatrickFu0302/time-series-feature-demo/main/'

@st.cache_data
def read_data():
    pca_matrix = pd.read_csv(os.path.join(path_data, 'pca_matrix.csv'))
    tsne_matrix = pd.read_csv(os.path.join(path_data, 'tsne_matrix.csv'))
    return pca_matrix, tsne_matrix


data_PCA, data_tSNE_2d = read_data()

#%% Sidebar for filters
st.sidebar.markdown('## Locations')
selected_Locations = st.sidebar.multiselect(
    label = 'Select Locations for visualization:',
    options = list(data_PCA['Location'].unique()),
    default = list(data_PCA['Location'].unique()))

st.sidebar.markdown('## Type')
selected_bldg_types = st.sidebar.multiselect(
    label = 'Select building types for visualization:',
    options = list(data_PCA['Type'].unique()),
    default = list(data_PCA['Type'].unique()))

st.sidebar.markdown('## Transparency')
alpha = st.sidebar.slider(label = 'Select transparency of dots in scatter plots:', 
min_value=0.1, max_value=1.0, value=0.5)

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    try:
        df.drop('Unnamed: 0',axis=1,inplace=True)
    except:
        pass
    return df.to_csv(index=False).encode('utf-8')

csv_PCA = convert_df(data_PCA)
csv_tSNE = convert_df(data_tSNE_2d)

st.sidebar.markdown('## Download the features:')
st.sidebar.download_button(
    label="Download PCA data as CSV",
    data=csv_PCA,
    file_name='pca_matrix.csv',
)
st.sidebar.download_button(
    label="Download t-SNE data as CSV",
    data=csv_tSNE,
    file_name='tsne_matrix.csv',
)

st.sidebar.markdown('## Github repository: \n https://github.com/hussainkazmi')

    
#%% Filter data for visualization
pca_matrix = data_PCA[data_PCA['Location'].isin(selected_Locations)&\
                  data_PCA['Type'].isin(selected_bldg_types)]

tsne_matrix = data_tSNE_2d[data_tSNE_2d['Location'].isin(selected_Locations)&\
                          data_tSNE_2d['Type'].isin(selected_bldg_types)]
    

#%% Scatter plots of PCA results
st.markdown('#### Scatter plots of PCA results')

with st.expander("See the dataframe of PCA features"):
    st.markdown('Rows of dataframe: '+str(len(pca_matrix)))
    st.write(pca_matrix.reset_index(drop=True))
    
PCA_fig1 = plt.figure(figsize=(8, 6))
PCA_ax1 = sns.scatterplot(data=pca_matrix[pca_matrix['PC_0']<80], x="PC_0", y="PC_1", hue="Location", alpha=alpha)
sns.move_legend(PCA_ax1, "upper left", bbox_to_anchor=(1, 1))
st.pyplot(PCA_fig1)

PCA_fig2 = plt.figure(figsize=(8, 6))
PCA_ax2 = sns.scatterplot(data=pca_matrix[pca_matrix['PC_0']<80], x="PC_0", y="PC_1", hue="Type", alpha=alpha)
sns.move_legend(PCA_ax2, "upper left", bbox_to_anchor=(1, 1))
st.pyplot(PCA_fig2)

#%% Scatter plots of t-SNE results
st.markdown('#### Scatter plots of t-SNE results')

with st.expander("See the dataframe of t-SNE features"):
    st.markdown('Rows of dataframe: '+str(len(tsne_matrix)))
    st.write(tsne_matrix.reset_index(drop=True))
    
tsne_fig1 = plt.figure(figsize=(8, 6))
tsne_ax1 = sns.scatterplot(data=tsne_matrix, x="tSNE_x", y="tSNE_y", hue="Location", alpha=alpha)
sns.move_legend(tsne_ax1, "upper left", bbox_to_anchor=(1, 1))
st.pyplot(tsne_fig1)

tsne_fig2 = plt.figure(figsize=(8, 6))
tsne_ax2 = sns.scatterplot(data=tsne_matrix, x="tSNE_x", y="tSNE_y", hue="Type", alpha=alpha)
sns.move_legend(tsne_ax2, "upper left", bbox_to_anchor=(1, 1))
st.pyplot(tsne_fig2)
