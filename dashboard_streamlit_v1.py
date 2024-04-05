'''
In this part, we will build a interactive web app with data.
To construct the app, streamlit library will be used as it can help us building the app faster and easier.
'''

# importing necessary libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # plotly help us to build interactive chart

# setting page layout as wide to display
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# loading the cleaned CSV file data
animes = pd.read_csv('./anime_data_2006_2022_cleaned.csv')

# from here, we will build the side panel

# year slider
selected_year = st.sidebar.slider(
    'Select a year', min_value=2006, max_value=2022, value=2017)

custom_order = ['Winter', 'Spring', 'Summer', 'Fall']
options = sorted(animes['season'].unique(),
                 key=lambda x: custom_order.index(x))
container_season = st.sidebar.container()
all_season = st.sidebar.checkbox("Select all season", value=True)
if all_season:
    selected_season = container_season.multiselect(
        'Select Season', options, default=options)
else:
    selected_season = container_season.multiselect('Select Season', options)

# type selectbox
selected_type = st.sidebar.selectbox(
    'Select Type', animes['type'].unique()
)

# demographic checkbox
options = animes['demographics'].str.split(
    ',', expand=True).stack().str.strip().unique()
container_demographics = st.sidebar.container()
all_demographics = st.sidebar.checkbox('Select all demographic', value=True)
if all_demographics:
    selected_demographic = container_demographics.multiselect(
        'Select Demographic', options, default=options
    )
else:
    selected_demographic = container_demographics.multiselect(
        'Select Demographic', options
    )

# genre checkbox
options = sorted(animes['genres'].str.split(
    ',', expand=True).stack().str.strip().unique())
container_genre = st.sidebar.container()
all_genre = st.sidebar.checkbox("Select all genre", value=True)
if all_genre:
    selected_genre = container_genre.multiselect(
        # [option for option in options if option != 'Others']
        'Select Genre', options, default=options
    )
else:
    selected_genre = container_genre.multiselect(
        'Select Genre', options)

# theme checkbox
options = sorted(animes['themes'].str.split(
    ',', expand=True).stack().str.strip().unique())
container_theme = st.sidebar.container()
all_theme = st.sidebar.checkbox("Select all theme", value=True)
if all_theme:
    selected_theme = container_theme.multiselect(
        # [option for option in options if option != 'Others']
        'Select Theme', options, default=options
    )
else:
    selected_theme = container_theme.multiselect(
        'Select Theme', options
    )


def is_subset(value, check_list):
    value_list = value.split(',')
    return any(value in check_list for value in value_list)


# filtering the data according to the inputs from the sidebar
filtered_df = animes[
    (animes['year'] == selected_year) &
    (animes['season'].isin(selected_season)) &
    (animes['type'] == selected_type) &
    (animes['demographics'].isin(selected_demographic)) &
    (animes['genres'].apply(is_subset, args=(selected_genre,))) &
    (animes['themes'].apply(is_subset, args=(selected_theme,)))
]\
    .sort_values(by=['score', 'rank'], ascending=False)\
    .reset_index()

# setting title
st.title(f'Top 10 Animes ({selected_year})')
st.markdown(f'<p>{", ".join(selected_season)}</p>', unsafe_allow_html=True)

# displaying the filtered data
show_table_df = filtered_df[[
    'title',
    'score',
    'episodes',
    'source',
    'season',
    'genres',
    'themes',
    'demographics',
    'rating'
]].head(10)

# making two columns to show the filtered data table on the left and its images from each data on the right
col1, col2 = st.columns(2)

show_table_df.index += 1  # setting data table index to 1 as its default starts from 0
# left column
with col1:
    st.write(show_table_df)

# right column
with col2:
    images = [row['image_url']
              for index, row in filtered_df.head(10).iterrows()]
    st.image(images, width=120)


def get_category_count_from_str(col):
    cat_df = col.str.split(',', expand=True).stack().str.strip()
    cat_df = pd.get_dummies(cat_df, prefix='', prefix_sep='')

    cat_counts = cat_df.sum(axis=0)

    return cat_counts


# checking if there is filtered data or not
if filtered_df.shape[0]:
    # scatter plot
    genre_counts = get_category_count_from_str(
        filtered_df[filtered_df['genres'] != 'Others']['genres'])
    genre_df = pd.DataFrame(
        {'genre': genre_counts.index, 'count': genre_counts.values})
    genre_fig = px.scatter(genre_df, x='genre', y='count', size='count', title='Anime Genre Distribution',
                           labels={'count': 'Number of Anime', 'genre': 'Genre'}, color='genre', hover_name='genre', size_max=100, template="plotly_dark")
    genre_fig.update_xaxes(showticklabels=False, title=None)
    genre_fig.update_yaxes(showticklabels=False)
    genre_fig.update_layout(
        autosize=True,
        width=1200,  # Adjust figure width
        height=700  # Adjust figure height
    )
    st.plotly_chart(genre_fig, use_container_width=True)

    # polar plot
    theme_counts = get_category_count_from_str(
        filtered_df[filtered_df['themes'] != 'Others']['themes'])
    theme_df = pd.DataFrame(
        {'theme': theme_counts.index, 'count': theme_counts.values}
    )
    theme_fig = px.bar_polar(theme_df, theta='theme',
                             r='count', title='Anime Theme Distribution', color='count', template="plotly_dark",
                             color_discrete_sequence=px.colors.sequential.Plasma_r)
    theme_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
            angularaxis=dict(
                tickangle=0  # Adjust the rotation angle here
            )
        ),
        autosize=True,
        width=1200,  # Adjust figure width
        height=700,  # Adjust figure height
        font=dict(size=11),  # Adjust font size
    )
    st.plotly_chart(theme_fig, use_container_width=True)
