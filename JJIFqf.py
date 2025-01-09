'''

# reads in ranking list and allows selections for wild cards


'''
import numpy as np
from datetime import datetime

import re
from pandas import json_normalize
from fpdf import FPDF
import plotly.express as px


import pycountry_convert as pc
import streamlit as st

import plotly.graph_objs as pg
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# uri of sportdataAPI
BASEURI = "https://www.sportdata.org/ju-jitsu/rest/"


# some dictionaries for JJIF Colors
COLOR_MAP_CON = {"Europe": 'rgb(243, 28, 43)',
                 "Asia": 'rgb(0,144,206)',
                 "Pan America": 'rgb(211,211,211)',
                 "Africa": 'rgb(105,105,105)',
                 "Oceania": 'rgb(255,255,255)'}

class PDF(FPDF):
    '''
    overwrites the pdf settings
    might be needed later
    '''

    def __init__(self, orientation, tourname):
        # initialize attributes of parent class
        super().__init__(orientation)
        # initialize class attributes
        self.tourname = tourname

    def header(self):
        # Logo
        self.image('Logo_real.png', 8, 8, 30)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(70)
        # Title
        self.cell(30, 10, 'Seeding ' + self.tourname, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number & printing date
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cell(0, 10, 'Printed ' + str(now) + ' Page ' +
                  str(self.page_no()) + '/{nb}', 0, 0, 'C')


def read_in_iso():
    ''' Read in file
     - HELPER FUNCTION TO READ IN A CSV FILE and convert NOC code to ISO

    '''
    inp_file = pd.read_csv("Country,NOC,ISOcode.csv", sep=',')
    ioc_iso = inp_file[
        ['NOC', 'ISO code']
    ].set_index('NOC').to_dict()['ISO code']

    return ioc_iso


def read_in_catkey():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert category ids to category names

    '''
    inp_file = pd.read_csv('https://raw.githubusercontent.com/ClaudiaBehnke86/JJIFsupportFiles/main/catID_name.csv', sep=';')
    key_map_inp = inp_file[
        ['cat_id', 'name']
    ].set_index('cat_id').to_dict()['name']

    return key_map_inp


@st.cache_data
def get_standings(user, password):
    """
    get the athletes form sportdata per category & export to a nice data frame

    Parameters
    ----------
    rank_cat_id
        sportdata category_id (from ranking) [int]
    MAX_RANK_pos
        seeding will stop at this number [int]
    user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'/standings/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)

    d_in = response.json()
    df_out = json_normalize(d_in)


    # remove prefix (this is hard coded per event! )
    df_out['title'].replace("World Games 2025 Standing - ", " ", regex=True, inplace=True)
    cat_names = df_out["title"].to_list()

    urls = df_out["url"].to_list()

    list_df = []

    for i, url in enumerate(urls):
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df_in = df_list[-1]

        # only used relevant data
        df = df_in[['Lastname', 'Firstname', 'Country', 'Standing', 'Unnamed: 9_level_0']]
        # remove weird double column
        df.drop(('Country', 'TOP 10'), axis=1, inplace=True)
        # rename
        df = df.rename(columns={'Unnamed: 9_level_0': 'Points'})

        # flatten multilevel index
        df.columns = df.columns.get_level_values(0)
        # remove "TOP" entries
        df = df[~df['Country'].str.contains("TOP", na=False)]

        df['Standing'] = df['Standing'].astype(int)
        df['Category'] = cat_names[i]

        # make country codes
        df = df.rename(columns={'Country': 'country_code'})
        df['country_code'] = df['country_code'].str.split('(').str[1]
        df['country_code'] = df['country_code'].str.split(')').str[0]

        # convert neutral athletes into Liechtenstein
        # (make sure to change if we ever have a JJNO there)
        df["country_code"].replace("JJIF", "LIE", regex=True, inplace=True)
        df["country_code"].replace("JIF", "LIE", regex=True, inplace=True)
        df["country_code"].replace("AIN", "LIE", regex=True, inplace=True)

        # replace wrong country codes in data
        df["country_code"].replace("RJF", "RUS", regex=True, inplace=True)
        df["country_code"].replace("ENG", "GBR", regex=True, inplace=True)

        # convert IOC codes to ISO codes using a dict
        df['country_code'] = df['country_code'].replace(IOC_ISO)
        # set the continent
        df['continent'] = df['country_code'].apply(
            lambda x: pc.country_alpha2_to_continent_code(x))

        df['continent'] = df['continent'].apply(
            lambda x: pc.convert_continent_code_to_continent_name(x))

        df['Country'] = df['country_code'].apply(
            lambda x: pc.country_alpha2_to_country_name(x))
        df['country_code'] = df['country_code'].apply(lambda x: pc.country_alpha2_to_country_name(x))



        # some JJIF adaptions
        # we have a Pan American Union and not North and South Amerixa
        df['continent'].where(~(df['continent'].str.contains("South America")),
                                  other="Pan America", inplace=True)
        df['continent'].where(~(df['continent'].str.contains("North America")),
                                  other="Pan America", inplace=True)
        # ISR is part of the European Union
        df['continent'].where(~(df['country_code'].str.contains("ISR")),
                                  other="Europe", inplace=True)
        # TUR is part of the European Union
        df['continent'].where(~(df['country_code'].str.contains("TUR")),
                                  other="Europe", inplace=True)

        df['country_code'] = df['country_code'].apply(lambda x: pc.country_name_to_country_alpha3(x))
        df['Country'].replace("Taiwan, Province of China", "Chinese Taipei", regex=True, inplace=True)
        df['Country'].replace(",", "", regex=True, inplace=True)

        list_df.append(df)

    if len(list_df) > 0:
        df_rank = pd.concat(list_df)
    else:
        # just return empty dataframe
        df_rank = pd.DataFrame()

    return df_rank


def write_session_state():

    st.session_state["df_standings"] = df_standings
    st.session_state['dropcats'] = dropcats

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


def draw_as_table(df_in):
    ''' draws a dataframe as a table and then as a fig.
    Parameters
    ----------
    val
        value to be looked up
    dict
        dict that contains the keys and value

    '''

    header_color = 'grey'
    row_even_color = 'lightgrey'
    row_odd_color = 'white'

    fig_out = go.Figure(data=[go.Table(
                        columnwidth=[15, 40, 20, 25, 25, 20],
                        header=dict(values=["Position", "Name", "Country", "Ranking Position", "Ranking Points", "Similarity"],
                                    fill_color=header_color,
                                    font=dict(family="Arial", color='white', size=12),
                                    align='left'),
                        cells=dict(values=[df_in.position, df_in.name, df_in.country_code, df_in.ranking, df_in.totalpoints, df_in.similarity],
                                   line_color='darkslategray',
                                   # 2-D list of colors for alternating rows
                                   fill_color=[[row_odd_color, row_even_color]*2],
                                   align=['left', 'left', 'left', 'left', 'left'],
                                   font=dict(family="Arial", color='black', size=10)
                                   ))
                        ])

    numb_row = len(df_in.index)

    fig_out.update_layout(
        autosize=False,
        width=750,
        height=(numb_row+1) * 35,
        margin=dict(
            l=20,
            r=50,
            b=0,
            t=0,
            pad=4
            ),
        )

    return fig_out

# main program starts here

st.sidebar.image("https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
                 use_container_width=True)

password = st.sidebar.text_input("Enter the password")

# simple password
if password == st.secrets['application_pass']:

    # read in ISO for country display
    IOC_ISO = read_in_iso()

    # initialize session state if not yet there
    uploaded_file = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=False)
    if uploaded_file is not None:
        # remove index column
        df_standings = pd.read_csv(uploaded_file)

    elif 'df_standings' not in st.session_state:
        with st.spinner('Read in data'):
            df_standings = get_standings(st.secrets['user'], st.secrets['password'])
        st.session_state.df_standings = df_standings

        # add QF type
        df_standings['QF_type'] = None

        # select the top 4
        df_top4 = df_standings[df_standings['Standing'] < 5]
        # Set QF type to R = Ranking
        df_standings['QF_type'][df_standings['Standing'] < 5] = "R"
        df_top4['QF_type'] = "R"

    else:
        # get session state of df standing
        df_standings = st.session_state["df_standings"]


    df_top4 = df_standings[df_standings['Standing'] < 5]
    # only one athlete per JJNO is allowed per category
    df_doubleathletes = df_top4[df_top4.duplicated(subset=['Category', 'country_code'], keep=False)]

    # those are the double athletes, which needs to be "deselected"
    remove_df = df_top4[df_top4.duplicated(subset=['Category', 'country_code'], keep='first')]

    # update standings
    df_standings = pd.merge(df_standings, remove_df, on=['Category', 'country_code', 'Country', 'Firstname', 'Lastname', 'continent', 'Standing', 'Points', 'QF_type'], how='left', indicator='Double')

    # remove R at double nations
    df_standings.loc[df_standings['Double'] == "both", "QF_type"] = None

    for cat in df_standings['Category'].unique():
        df_cut_cat = df_standings[(df_standings['Category']==cat)]
        cur_len = len(df_cut_cat[df_cut_cat['QF_type'] == "R"])
        index = 5 - cur_len
        while cur_len < 4:
            if (index + cur_len) < len(df_cut_cat):
                # jjnos in category
                jjnos_cat = df_cut_cat['country_code'][df_cut_cat['QF_type'] == "R"].unique().tolist()
                # loop over next athletes
                df_next_ath = df_cut_cat[(~df_cut_cat['country_code'].isin(jjnos_cat)) & (df_cut_cat['Standing'] == (index + cur_len))]
                df_top4 = pd.concat([df_top4, df_next_ath])
                #
                df_standings['QF_type'][(~df_standings['country_code'].isin(jjnos_cat)) &(df_standings['Category']==cat)&(df_standings['Standing'] == (index + cur_len))] = "R"
                df_top4['QF_type'][(df_top4['Category']==cat)&(df_top4['Standing'] == (index + cur_len))] = "R"
                index = index + 1
                cur_len = len(df_standings[(df_standings['Category']==cat) & (df_standings['QF_type'] == "R" )])
            else:
                st.warning(f'{cat} does not have enough athletes for ranking', icon="⚠️")
                cur_len = 4

    # Clean indicators
    del df_top4['Double']
    del df_standings['Double']

    # make all countries to a list
    all_countries = df_standings['Country'].unique().tolist()

    # make a list if all categories
    cat_list = df_standings['Category'].unique()

    # create tabs
    select, categories, countries, graphics = st.tabs(["Select wildcards", "Categories", "Countries","Show Graphics"])

    if 'dropcats' not in st.session_state:
        dropcats = []
        st.session_state["dropcats"] = dropcats
    else:
        # get session state
        dropcats = st.session_state["dropcats"]

    with select:
        st.header("Select Wild Cards")

        # drop categories which have 6 athletes selected
        number_of_options = 102 - len(df_standings[df_standings['QF_type'].notnull()])
        if number_of_options > 0:

            st.session_state.df_standings = df_standings

            # make a list of all qf countries
            selected_countries = df_standings['Country'][df_standings['QF_type'].notnull()].unique().tolist()
            # countries without athletes in the top 4 but athletes QF
            wc_countries = [x for x in all_countries if x not in selected_countries]

            # make a list of all categories countries

            # select those which can be selectable
            df_selectable = df_standings[df_standings['Country'].isin(wc_countries) & (~df_standings['Category'].isin(dropcats))]
            # sort
            df_selectable = df_selectable.sort_values(by=['Standing'])

            output_dict = st.dataframe(
                df_selectable[['Standing', 'Category', 'Country', 'Lastname', 'Firstname', 'Points']],
                use_container_width=True,
                hide_index=True,
                key="round"+str(number_of_options),
                on_select="rerun",
                selection_mode="multi-row"
            )
            selected_athletes = output_dict.selection.rows
            filtered_df = df_selectable.iloc[selected_athletes]
            df_standings = pd.merge(df_standings, filtered_df, on=['Category', 'country_code', 'Country', 'Firstname', 'Lastname', 'continent', 'Standing', 'Points', 'QF_type'], how='left', indicator='Double')
            # add Selection
            df_standings.loc[df_standings['Double'] == "both", "QF_type"] = "WC"
            del df_standings['Double']

            for cat in cat_list:
                if cat not in dropcats:
                    if len(df_standings[(df_standings['QF_type'].notnull()) & (df_standings['Category']==cat)]) >= 6:
                        st.warning(f'{cat} is full', icon="⚠️")
                        dropcats.append(cat)

            if st.button("Submit", on_click=write_session_state):

                # create download
                csv = convert_df(df_standings)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="selection.csv",
                    mime="text/csv",
                )
            st.write("Full categories ", str(dropcats))
            st.header("Wild Card Athletes")
            st.dataframe(df_standings[df_standings['QF_type']== "WC"], use_container_width=True, hide_index=True, column_order=['Category','Standing', 'Country', 'Lastname', 'Firstname', 'Points'])

    with graphics:
        # some pics

        # countries with athletes in the top 4
        df_standings_points_sel = df_standings[(df_standings['QF_type'].notnull())]
        df_top4_fig = df_standings_points_sel[['Country', 'Standing', 'Firstname']].groupby(['Country', 'Standing']).count().reset_index()
        fig1 = px.bar(df_top4_fig, x='Country', y='Firstname',
                      color='Standing', title="QF countries",
                      labels={
                                "Firstname": "Number of Athletes",
                                }
                      )
        fig1.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig1)

        # selection
        df_standings_sel = df_standings[(df_standings['QF_type'].notnull())]
        df_standings_fig = df_standings_sel[['Country', 'Standing', 'Firstname', 'QF_type']].groupby(['Country', 'Standing', 'QF_type']).count().reset_index()
        fig1 = px.bar(df_standings_fig, x='Country', y='Firstname',
                      color='QF_type', title="QF countries",
                      labels={
                                "Firstname": "Number of Athletes",
                                }
                      )
        fig1.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig1)

        # show a map with selected athletes
        df_map1 = pd.DataFrame()
        df_map1['country'] = df_standings['country_code'][df_standings['QF_type'].notnull()].value_counts().index
        df_map1['counts'] = df_standings['country_code'][df_standings['QF_type'].notnull()].value_counts().values

        data = dict(type='choropleth',
                    locations=df_map1['country'], z=df_map1['counts'])

        layout = dict(title='Athletes in QF',
                      geo=dict(showframe=True,
                               projection={'type': 'robinson'}))
        x = pg.Figure(data=[data], layout=layout)
        x.update_geos(
                showcountries=True, countrycolor="black"
        )
        st.plotly_chart(x)

        df_jjcu = pd.DataFrame()
        df_jjcu['continent'] = df_standings['continent'][df_standings['QF_type'].notnull()].value_counts().index
        df_jjcu['counts'] = df_standings['continent'][df_standings['QF_type'].notnull()].value_counts().values
        fig2 = px.pie(df_jjcu, values='counts', names='continent',
                      color='continent', color_discrete_map=COLOR_MAP_CON,
                      title='continent distribution total')
        st.plotly_chart(fig2, use_container_width=True)

        left_column, right_column = st.columns(2)
        with left_column:
            df_jjcu_R = pd.DataFrame()
            df_jjcu_R['continent'] = df_standings['continent'][df_standings['QF_type']== "R"].value_counts().index
            df_jjcu_R['counts'] = df_standings['continent'][df_standings['QF_type']== "R"].value_counts().values
            fig_R = px.pie(df_jjcu_R, values='counts', names='continent',
                          color='continent', color_discrete_map=COLOR_MAP_CON,
                          title='continent distribution Ranking')
            st.plotly_chart(fig_R, use_container_width=True)

        with right_column:
            df_jjcu_WC = pd.DataFrame()
            df_jjcu_WC['continent'] = df_standings['continent'][df_standings['QF_type']== "WC"].value_counts().index
            df_jjcu_WC['counts'] = df_standings['continent'][df_standings['QF_type']== "WC"].value_counts().values
            fig_WC = px.pie(df_jjcu_WC, values='counts', names='continent',
                          color='continent', color_discrete_map=COLOR_MAP_CON,
                          title='continent distribution Wild Cards')
            st.plotly_chart(fig_WC, use_container_width=True)


    with categories:
        for cat in cat_list:
            st.header(cat)

            st.write("via Ranking")
            st.dataframe(df_standings[(df_standings['QF_type']== "R") & (df_standings['Category']==cat)], use_container_width=True, hide_index=True, column_order=['Standing', 'Country', 'Lastname', 'Firstname', 'Points'])

            st.write("via Wild Card")
            st.dataframe(df_standings[(df_standings['QF_type']== "WC") & (df_standings['Category']==cat)], use_container_width=True, hide_index=True, column_order=['Standing', 'Country', 'Lastname', 'Firstname', 'Points'])

            st.write("next in Ranking")
            st.dataframe(df_standings[(df_standings['QF_type'].isnull()) & (df_standings['Standing'] < 10) & (df_standings['Category']==cat)], use_container_width=True, hide_index=True, column_order=['Standing', 'Country', 'Lastname', 'Firstname', 'Points'])

    with countries:
        st.write('Qualfied countries: ', len(df_standings['Country'][df_standings['QF_type'].notnull()].unique().tolist()))
        st.write('Total countries: ', len(df_standings['Country'].unique().tolist()))

        for country in all_countries:
            if len(df_standings[(df_standings['QF_type'].notnull()) & (df_standings['Country']==country)]) > 0:
                st.header(country)
                if len(df_top4[df_top4['Country'] == country]) > 0:
                    st.write("Qualified via Ranking")
                    st.dataframe(df_top4[df_top4['Country']==country], use_container_width=True, hide_index=True, column_order=['Lastname', 'Firstname', 'Category','Standing'])

                if len(df_doubleathletes[df_doubleathletes['Country'] == country]) > 0:
                    st.write("Double Athletes")
                    st.dataframe(df_doubleathletes[df_doubleathletes['Country']==country], use_container_width=True, hide_index=True, column_order=['Lastname', 'Firstname', 'Category','Standing'])

                if len(df_standings[(df_standings['QF_type'] == "WC") & (df_standings['Country']==country)]) > 0:
                    st.write("Wild Card Athletes")
                    st.dataframe(df_standings[(df_standings['QF_type'] == "WC") & (df_standings['Country']==country)], use_container_width=True, hide_index=True, column_order=['Lastname', 'Firstname', 'Category','Standing'])

                if len(df_standings[(df_standings['Standing'] < 8) & (df_standings['Country']==country)]) > 0:
                    st.write("List of replacements")
                    st.dataframe(df_standings[(df_standings['QF_type'].isnull()) &(df_standings['Standing'] < 10) & (df_standings['Standing'] > 4) & (df_standings['Country']==country)], use_container_width=True, hide_index=True, column_order=['Lastname', 'Firstname', 'Category','Standing'])

else:
    st.image(
            "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExam11YTN6em8xcnpvaGk2eWYycHUxMHpzMWw2NDh5azRycndxOHVheiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xUStFKHmuFPYk/giphy.webp",
        )

st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
