import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

animes = pd.read_csv('./anime_data_2006_2022_cleaned.csv')

date_pattern = r'(\w{3} \d{1,2}, \d{4})'
animes['start_date'] = animes['aired_date'].str.extract(date_pattern)
animes['start_date'] = pd.to_datetime(animes['start_date'])


def get_season(month):
    if 4 <= month <= 6:
        return 'Spring'
    elif 7 <= month <= 9:
        return 'Summer'
    elif 10 <= month <= 12:
        return 'Fall'
    else:
        return 'Winter'


# Apply the function to create a new 'season' column
animes['season'] = animes['start_date'].dt.month.apply(get_season)

animes['title'] = np.where(animes['title_english'].notnull(),
                           animes['title_english'],
                           animes['title']
                           )

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


selected_type = st.sidebar.selectbox(
    'Select Type', animes['type'].unique()
)

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

st.title(f'Top 10 Animes ({selected_year})')
st.markdown(f'<p>{", ".join(selected_season)}</p>', unsafe_allow_html=True)

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


col1, col2 = st.columns(2)
show_table_df.index += 1
with col1:
    st.write(show_table_df)

with col2:
    images = [row['image_url']
              for index, row in filtered_df.head(10).iterrows()]
    st.image(images, width=120)


def get_category_count_from_str(col):
    cat_df = col.str.split(',', expand=True).stack().str.strip()
    cat_df = pd.get_dummies(cat_df, prefix='', prefix_sep='')

    cat_counts = cat_df.sum(axis=0)

    # print(cat_counts.sort_values(ascending=False))
    return cat_counts


if filtered_df.shape[0]:
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
        width=1200,
        height=700
    )
    # with col1:
    st.plotly_chart(genre_fig, use_container_width=True)

    theme_counts = get_category_count_from_str(
        filtered_df[filtered_df['themes'] != 'Others']['themes'])
    theme_df = pd.DataFrame(
        {'theme': theme_counts.index, 'count': theme_counts.values}
    )
    # st.write(theme_df)
    # theme_fig = px.funnel(theme_df.sort_values(
    #     by='count'), y='theme', x='count')
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
        font=dict(size=11),
    )
    # with col2:
    st.plotly_chart(theme_fig, use_container_width=True)

# st.write("Anime Details")
# for index, row in filtered_df.iterrows():
#     st.image(row['image_url'], caption=row['title'])
# st.write('''
#     <style>
#         .text-color {
#             color: red;
#         }
#     </style>
# ''', unsafe_allow_html=True)

# st.write("Anime List")
# st.write(
#     "<style>table {border-collapse: collapse;} table, th, td {border: 1px solid black; padding: 8px;}</style>", unsafe_allow_html=True)
# st.write("<table><tr><th>Title</th><th>Image</th><th>Trailer</th></tr>",
#          unsafe_allow_html=True)

# for index, row in filtered_df.head(10).iterrows():
#     st.write(
#         f"<tr><td style='width: 200px;'><a href={row['url']} target='_blank'>{row['title']}</a></td><td><img src='{row['image_url']}' style='width: 150px; height: 200px;'></td><td><iframe width='560' height='315' src='https://www.youtube.com/embed/{row['trailer_url'].split('=')[-1] if isinstance((row['trailer_url']), str) else ''}' frameborder='0' allowfullscreen></iframe></td></tr>", unsafe_allow_html=True)
# st.write("</table>", unsafe_allow_html=True)
# st.write("<p class='text-color'>Hello World!</p>", unsafe_allow_html=True)

# st.markdown('''
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             background-color: #f4f4f4;
#             margin: 0;
#             padding: 0;
#         }
#         h1 {
#             text-align: center;
#             margin-top: 20px;
#         }
#         table {
#             width: 80%;
#             margin: 20px auto;
#             border-collapse: collapse;
#             box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
#             background-color: #fff;
#         }
#         th, td {
#             padding: 15px;
#             text-align: left;
#         }
#         th {
#             background-color: #333;
#             color: #fff;
#         }
#         tr:nth-child(even) {
#             background-color: #f2f2f2;
#         }
#         iframe {
#             width: 100%;
#             height: 200px;
#         }
#     </style>
# ''', unsafe_allow_html=True)

# st.write('''
#     <h1>Anime Info Table</h1>
#     <table>
#         <tr>
#             <th>Title</th>
#             <th>Image</th>
#             <th>Trailer</th>
#         </tr>
#     <tr>
# ''', unsafe_allow_html=True)

# for index, row in filtered_df.head(10).iterrows():
#     st.write(f"<tr><td>{row['title']}</td><td><img src='{row['image_url']}'></td><td><iframe src='https://www.youtube.com/embed/{row['trailer_url'].split('=')[-1] if isinstance((row['trailer_url']), str) else ''}' frameborder='0' allowfullscreen></iframe></td></tr>", unsafe_allow_html=True)

# st.write("</table>", unsafe_allow_html=True)
