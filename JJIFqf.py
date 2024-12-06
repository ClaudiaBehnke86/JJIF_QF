'''

Reads in the registrations and makes a seeding based on the ranking list

Names are mapped using:
https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e
force A and B as a CSR matrix.
With sportse_dot_topn updated https://github.com/ing-bank/sparse_dot_topn/tree/master


'''
import numpy as np
from datetime import datetime
import os
import re
from pandas import json_normalize
from fpdf import FPDF

import streamlit as st
import plotly.graph_objects as go

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# for the name matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

class PDF(FPDF):
    '''
    overwrites the pdf settings
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


def read_in_cat_rankid():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert category ids to category names

    '''
    inp_file = pd.read_csv("catID_rankID.csv", sep=';')
    rank_cat_id_out = inp_file[
        ['cat_id', 'ranking_id']].set_index('cat_id').to_dict()['ranking_id']

    return rank_cat_id_out


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

# uri of sportdataAPI
BASEURI = "https://www.sportdata.org/ju-jitsu/rest/"


# since teams categories have no country I use this quick and dirty workaround
# to map clubnames in sportdata api to country codes...
CLUBNAME_COUNTRY_MAP = {"Belgian Ju-Jitsu Federation": 'BEL',
                        "Deutscher Ju-Jitsu Verband e.V.": 'GER',
                        "Federazione Ju Jitsu Italia": 'ITA',
                        "Team Italia":'ITA',
                        "Romanian Martial Arts Federation": 'ROU',
                        "Österreichischer Jiu Jitsu Verband": "AUT",
                        "Taiwan Ju Jitsu Federation": "TPE",
                        "Royal Spain Ju Jutsi Federation": 'ESP',
                        "Federation Française de Judo, Jujitsu, Kendo et DA": 'FRA',
                        "Pakistan Ju-Jitsu Federation": 'PAK',
                        "Vietnam Jujitsu Federation": 'VIE',
                        "Judo Bond Nederland": "NED",
                        "Polish Sport Ju-Jitsu Association":"POL",
                        "Ju-Jitsu Federation of Slovenia":'SLO',
                        "Hellenic Ju-Jitsu Federation": 'GRE',
                        "Ju Jitsu Association of Thailand": 'THA',
                        "FÉDÉRATION ROYALE MAROCAINE DE JU-JITSU ": 'MAR',
                        "Ju-Jitsu Federation of Montenegro": 'MNE',
                        "Swiss Judo & Ju-Jitsu Federation": 'SUI',
                        "Kazakhstan Jiu Jitsu Association": 'KAZ',
                        "Mongolian Jiu-Jitsu Association": 'MGL',
                        "Algerian Federation of Ju Jitsu": 'ALG',
                        "Federacion Colombiana de Jiu-Jitsu": 'COL',
                        "Federaciòn Costarricense de Jiu Jitsu": 'CRC',
                        "Federaciòn Uruguaya de Jiu Jitsu": 'URU',
                        "Asociatiòn Argentina de Jiu Jitsu": 'ARG',
                        "Bulgarian Ju-Jitsu Federation": 'BUL',
                        "Bangladesh Ju-Jitsu Association": 'BAN',
                        "TUNISIAN JUJITSU FEDERATION": 'TUN'
                        }


def conv_to_type(df_in, type_name, type_list):
    '''
    checks strings in data frames and
    replaces them with types based on the _INP lists (see line 28 - 49)

    Parameters
    ----------
    df_in
        data frame to check [str]
    type_name
        the name of the _INP list to check [str]
    type_list
        of the _INP list to check [list]
    '''
    for inp_str in type_list:
        df_in[type_name].where(~(df_in[type_name].str.contains(inp_str)),
                               other=inp_str, inplace=True)

    return df_in[type_name]


def ngrams(string, n_gr=3):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    used to check for similar names
    Parameters
    ----------
    string
        input string
    n_gr
        ?

    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams_in = zip(*[string[i:] for i in range(n_gr)])
    return [''.join(ngram_in) for ngram_in in ngrams_in]


def get_matches_df(sparse_matrix, name_vector, top=100):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    unpacks the resulting sparse matrix
    '''
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if sparsecols.size > top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity_in = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similarity_in[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similarity': similarity_in})


@st.cache_data
def get_participants(eventid, user, password):
    """
    get the athletes form sportdata export to a nice data frame

    Parameters
    ----------
    eventid
        sportdata event_id (from database) [int]

     user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'event/members/'+str(eventid)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()

    # check if df is filled
    if len(d_in) > 0:
        df_out = json_normalize(d_in)
    else:
        df_out = pd.DataFrame()
        return df_out

    # drop rows that have no 'categories' (e.g. VIP's)
    if 'categories' in df_out.columns:
        df_out.dropna(inplace=True, subset="categories")
    else:
        return pd.DataFrame()

    # create a temporary column that counts how many categories an athlete has
    # participated in
    df_out["no_of_categories"] = df_out["categories"].apply(lambda x: len(x))
    # duplicate each row n times based on the number (n) of categories
    df_out = df_out.loc[df_out.index.repeat(df_out["no_of_categories"])]
    # create a temporary integer index, this is now the only variable that
    # differentiates the rows that were copied in the previous line
    df_out["temp_index"] = np.arange(0, len(df_out))
    # now groupby the unique id, make a transform and assign each row an
    # iterator that counts the occurence of duplicate rows
    df_out["category_counter"] = df_out.groupby(
        "id"
    )["temp_index"].transform(lambda x: x - x.min())
    # now we can use this 'category' to unfold the categories dictionaries
    # do this in a loop over the max of category_counter to ensure that
    # only vectorized operations are used and we don't need a nested 'apply'
    df_out["categories_unique"] = ""
    df_out["cat_id"] = 0

    # reserve an empty series that contains the categories that an
    # athlete participates in. we create a separate series for athletes
    # that participate only in 1 category, athletes that participate
    # in 2 categories, etc (i_cat is the iterator for this)
    i_cat_series = pd.Series()
    for i_cat in range(0, df_out["category_counter"].max() + 1):
        _temp_series = df_out[
            df_out["category_counter"] == i_cat
        ]["categories"].apply(lambda x: x[i_cat]["id"])
        i_cat_series = pd.concat([i_cat_series, _temp_series])

    # assign, for each athlete in the df_out dataframe, the category
    # or categories that they have participated in
    df_out["cat_id"] = i_cat_series.sort_index()

    # fit names to right formate
    df_out['name'] = df_out['first'] + " " + df_out['last']

    # clean up leading & trailing whitespaces and remove double
    df_out['name'].replace(r"^ +| +$", r"", regex=True, inplace=True)
    df_out['name'].replace("  ", " ", regex=True, inplace=True)

    # clean up unused and temporary columns
    del df_out["categories"]
    del df_out["categories_unique"]
    del df_out["temp_index"]
    del df_out["category_counter"]
    del df_out["no_of_categories"]
    del df_out["country_name"]
    del df_out["flag"]
    del df_out["club_id"]
    del df_out["club_name"]
    del df_out["dateofbirth"]
    del df_out["passport_id"]
    del df_out["type"]
    del df_out["sex"]

    return df_out


@st.cache_data
def get_couples(eventid, user, password):
    """
    get the athletes form sportdata per category & export to a nice data frame

    Parameters
    ----------
    eventid
        sportdata event_id (from database) [int]
    cat_id
        sportdata category_id (from database) [int]
     user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'event/categories/'+str(eventid)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d = response.json()

    if 'categories' in d:
        df_out = json_normalize(d['categories'])

        num_par = df_out[['id', 'number_of_participants']]

    else:
        df_ath_couples = pd.DataFrame()
        return df_ath_couples

    duo_ids = [1491, 1492, 21351, 1490, 1889, 1890, 1891, 1488, 1487, 1489]
    show_ids =[1494, 1493, 21185, 1495, 1892, 1893, 1894, 1497, 1498, 1496]

    couple_ids = duo_ids + show_ids

    num_par = num_par[num_par["id"].isin(couple_ids)]

    list_df_couples = []

    for cat_id in num_par["id"].tolist():

        uri = str(BASEURI)+'event/categories/'+str(eventid)+'/'+str(cat_id)+'/'

        response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
        d_in = response.json()
        if len(d_in) > 0:
            df_out = json_normalize(d_in["members"])
        else:
            df_out = pd.DataFrame()

        if not df_out.empty:
            # for an unclear reason teams to no have a country code...
            # convert club name to country using dict...
            df_out['country_code'] = df_out['club_name'].replace(CLUBNAME_COUNTRY_MAP)

            df_err = df_out.loc[df_out['country_code'].str.len() > 3]
            if len(df_err) > 0:
                st.error("There are non converted club names")
                st.write("send e-mail to sportdirector@jjif.org with the following details:")
                st.write(df_out.loc[df_out['country_code'].str.len()> 3])
            df_out['name'] = df_out['name'].str.split('(').str[1]
            df_out['name'] = df_out['name'].str.split(')').str[0]
            df_out['name'].replace(",", " ", regex=True, inplace=True)
            df_out['name'].replace("_", " ", regex=True, inplace=True)
            df_ath = df_out[['name', 'country_code']]
            df_ath['cat_id'] = cat_id
            df_ath = df_ath.astype(str)
        else:
            # just return empty dataframe
            df_ath = pd.DataFrame()

        list_df_couples.append(df_ath)

    if len(list_df_couples) > 0:
        df_ath_couples = pd.concat(list_df_couples)
    else:
        # just return empty dataframe
        df_ath_couples = pd.DataFrame()

    return df_ath_couples


def get_event_name(eventid, user, password):
    """
    get the event name from sportdata as string

    Parameters
    ----------
    eventid
        sportdata event_id (from database) [int]
     user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'/event/'+str(eventid)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_out = json_normalize(d_in)
    event_name = df_out['name'].astype(str)[0]
    return event_name


def get_ranking_cat(user, password):
    """
    ranking has different category ids...
    so get a dict with them
    """

    # URI of the rest API
    uri = str(BASEURI)+'/ranking/categories/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_rankcats = json_normalize(d_in)
    df_rankcats = df_rankcats.drop(['cat_sex', 'cat_isteam'], axis=1)
    df_rankcats = df_rankcats.set_index('cat_id')
    my_series = df_rankcats['cat_title'].squeeze()
    dict_ranking = my_series.to_dict()
    return dict_ranking


@st.cache_data
def get_ranking(rank_cat_id, max_rank_pos, user, password):
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
    uri = str(BASEURI)+'/ranking/category/'+str(rank_cat_id)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_out = json_normalize(d_in)

    if not df_out.empty:
        df_out['name'] = df_out['name'].str.split('(').str[0]
        df_out['name'] = df_out['name'].str.rstrip()
        df_rank = df_out[['name', 'countrycode', 'rank', 'cat_id', 'totalpoints', 'cat_title']]
        df_rank = df_rank.rename(columns={"countrycode": 'country_code'})
        # rename rank to ranking since df.rank is a function name
        df_rank['ranking'] = df_rank['rank'].astype(int)
        df_rank = df_rank[df_rank['ranking'] < int(max_rank_pos)]
        df_rank['ranking'] = df_rank['ranking'].astype(str)
    else:
        # just return empty datafram
        df_rank = pd.DataFrame()
    return df_rank


def rev_look(val_in, dict_in):
    ''' revese lookup of key.
    Returns first matching key
    Parameters
    ----------
    val
        value to be looked up
    dict
        dict that contains the keys and value

    '''
    key_out = next(key for key, value in dict_in.items() if value == val_in)

    return key_out


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
                 use_column_width='always')

# get upcoming events
uri_upc = "https://www.sportdata.org/ju-jitsu/rest/events/upcoming/"

# read in categories
response = requests.get(uri_upc, auth=HTTPBasicAuth(st.secrets['user'], st.secrets['password']), timeout=5)
d_upc = response.json()
df_upc = json_normalize(d_upc)

# skip events which are not tournaments
df_upc = df_upc[~df_upc['name'].str.contains("Referee")]
df_upc = df_upc[~df_upc['name'].str.contains("Course")]
df_upc = df_upc[~df_upc['name'].str.contains("REFEREE")]

offmail = ["sportdata@jjif.org","worlds@jjif.org","pesk@adsys.gr" ,"rick.frowyn@jjeu.eu", "office@jjau.org", "mail@jjif.org", "jjif@sportdata.org", "jiujitsucolombia@hotmail.com", "fjjitalia@gmail.com"]
evts = df_upc['name'][df_upc['contactemail'].isin(offmail)].tolist()
evts.append('Other')
option = st.sidebar.selectbox("Choose your event", evts,
                              help='if the event is not listed choose Other')

if option == 'Other':
    sd_key = st.sidebar.number_input("Enter the Sportdata event number",
                                     help='the number behind vernr= in the URL', value=325)
else:
    sd_key = int(df_upc['id'][df_upc['name'] == option])

tourname = get_event_name(str(sd_key), st.secrets['user'], st.secrets['password'])

st.title('Seeding for ' + tourname)

mode = st.sidebar.selectbox('Select mode', ['Top20', 'Top10'])

if mode == 'Top10':
    MAX_RANK = 10
else:
    MAX_RANK = 20

# preselected age_divisions
AGE_SEL = ["U18", "U21", "Adults"]

age_select = st.sidebar.multiselect('Select the age divisions for seeding',
                                    AGE_SEL,
                                    ["U21", "Adults"])

# ID_TO_NAME = read_in_catkey()
catID_to_rankID = read_in_cat_rankid()

# maps ids to human readable names
key_map = read_in_catkey()


# create empty temporary list for categories to merge into team categories
list_df_athletes = []
list_df_ranking = []

dict_ranking_ids = get_ranking_cat(st.secrets['user'], st.secrets['password'])

my_bar = st.progress(0)
with st.spinner('Read in data'):

    df_athletes_in = get_participants(str(sd_key),
                                      st.secrets['user'],
                                      st.secrets['password'])

    df_couples = get_couples(str(sd_key),
                             st.secrets['user'],
                             st.secrets['password'])

    if(len(df_athletes_in) > 0 and len(df_couples) >0):
        df_athletes = pd.concat([df_athletes_in, df_couples])
    elif len(df_athletes_in) > 0:
        df_athletes = df_athletes_in
    elif len(df_couples) > 0:
        df_athletes = df_couples
    else:
        df_athletes = []

    list_df_ath = []

    if len(df_athletes) > 0:
        df_athletes.dropna(inplace=True, subset="cat_id")
        df_athletes['cat_id'] = df_athletes['cat_id'].astype(int)
        # select the age divisions you want to seed
        df_athletes['cat_name'] = df_athletes['cat_id'].replace(key_map)
        if len(df_athletes['cat_id'][df_athletes['cat_id'] == df_athletes['cat_name'] ]) >0:
            with st.expander('Cat_ids without matching name', expanded=False):
                st.write(df_athletes['cat_id'][df_athletes['cat_id'] == df_athletes['cat_name'] ].unique().tolist())
                st.write('If you miss categories in your seeding send mail to sportdirector@jjif.org and mention the above numbers')

        df_athletes = df_athletes[df_athletes['cat_id'] != df_athletes['cat_name'] ]

        df_athletes['age_division'] = df_athletes['cat_name']
        df_athletes['age_division'] = conv_to_type(df_athletes, 'age_division', AGE_SEL)

        df_athletes['rank_id'] = df_athletes['cat_id'].replace(catID_to_rankID)
        df_athletes = df_athletes.astype(str)

        df_athletes = df_athletes[df_athletes['rank_id'] != "NONE"]

        # remove categories without ranking
        if len(df_athletes['cat_name'][df_athletes['cat_id'] == df_athletes['rank_id'] ]) > 0:
            with st.expander('Categories without ranking list', expanded=False):
                st.write(df_athletes['cat_name'][df_athletes['cat_id'] == df_athletes['rank_id'] ].unique().tolist())
                st.write('If you miss categories in your seeding send mail to sportdirector@jjif.org and mention the above numbers')

        df_athletes = df_athletes[df_athletes['cat_id'] != df_athletes['rank_id'] ]

        # remove what is not selected
        df_athletes = df_athletes[df_athletes['age_division'].isin(age_select)]

        # only read in rankings associated to the df_athletes
        df_athletes = df_athletes.sort_values(by=['cat_name'])
        # String comparison does not handle "+"" well... replaced with p in .csv
        # and here replaced back
        df_athletes['cat_name'].replace(" p", " +", regex=True, inplace=True)

        cat_list = df_athletes['rank_id'].unique()

        for j, key in enumerate(cat_list):
            ranking_cat = get_ranking(str(key),
                                      MAX_RANK,
                                      st.secrets['user'],
                                      st.secrets['password'])
            list_df_ranking.append(ranking_cat)
            my_bar.progress(0.5+((j+1)/len(cat_list))/2)

        with st.expander('Details on name matching', expanded=False):
            st.write('Similar names were matched to avoid missing mapping. This is based on:')
            st.write('https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e')
            col1, col2 = st.columns(2)
            with col1:
                min_team = st.number_input("Minimum value of similarity for Duo/Show", value=0.35, min_value=0.01, max_value=0.99)
            with col2:
                min_indi = st.number_input("Minimum value of similarity for individuals", value=0.55, min_value=0.01, max_value=0.99)
            st.warning("increase number means making the name matching stronger, decrease allows more fuzzy matches")

        if len(list_df_ranking) > 0:
            df_ranking = pd.concat(list_df_ranking)

            df_ranking['rank_id'] = df_ranking['cat_id']

            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

            # create dummy column for similarity score
            df_ranking['similarity'] = df_ranking['name']
            df_ranking['original_name'] = ''

            df_athletes['name'] = df_athletes['name'].str.upper()
            df_ranking['name'] = df_ranking['name'].str.upper()

            # loop over all ranks to match
            for cat in cat_list:

                # get the names of from leading dataframe (athletes) and ranking frame
                names_athletes = df_athletes[df_athletes['rank_id'] == cat]['name']
                names_ranking = df_ranking[df_ranking['rank_id'] == cat]['name']

                # skip category if empty
                if len(names_athletes) < 1:
                    df_ranking.loc[
                        df_ranking['rank_id'] == cat,
                        'similarity'
                    ] = None
                    continue

                # combine the two lists of names into one list
                all_names = pd.concat([names_athletes, names_ranking]).values
                # perform matching over combined name list

                matrix = vectorizer.fit_transform(all_names)
                if len(all_names) > 4:
                    matches = sp_matmul_topn(matrix,
                                             matrix.transpose(),
                                             top_n=9, threshold=0.3, sort=True)

                else:
                    matches = sp_matmul_topn(matrix,
                                             matrix.transpose(),
                                             top_n=4, threshold=0.4, sort=True)

                # create a dataframe with the matches
                df_matches = get_matches_df(matches, all_names)

                name_cat = df_athletes[df_athletes["rank_id"] == str(cat)]["cat_name"].tolist()


                # Duo names have much lower similarity
                if 'Duo' in name_cat[0]:
                    min_value = min_team
                elif 'Show' in name_cat[0]:
                    min_value = min_team
                else:
                    min_value = min_indi

                # remove self-mapping of names (exact matches)
                df_matches = df_matches[
                    (df_matches["similarity"] < .99999999) & (
                        df_matches["similarity"] > min_value)
                ]

                # sort by similarity to make sure the dic takes the right person
                df_matches = df_matches.sort_values(by=['similarity'])

                # create mapping dictionary for names
                dict_map = dict(zip(df_matches.left_side, df_matches.right_side))

                # if approximate matches are found, replace the names
                if len(dict_map) > 0:
                    # drop all keys in the mapping dict that do not have a value
                    # that appears in the athletes names. by doing this, you
                    # avoid replacing names in the ranking list with other names
                    # in the ranking list, which would potentially lead to mis-
                    # matches with athlete names
                    slim_dict = {key: value for key,
                        value in dict_map.items() if value in df_athletes[
                            df_athletes['rank_id'] == cat
                        ]['name'].values
                    }
                    similarity_dict = {
                        key: df_matches[
                            df_matches["right_side"] == key
                        ]['similarity'].values[0] for key, value in slim_dict.items()
                    }

                    # replace names in the ranking data
                    df_ranking.loc[
                        df_ranking['rank_id'] == cat,
                        'similarity'
                    ] = df_ranking[df_ranking['rank_id'] == cat]['similarity'].apply(lambda x:  similarity_dict.get(x))
                    df_ranking.loc[
                        df_ranking['rank_id'] == cat,
                        'original_name'
                    ] = df_ranking[df_ranking['rank_id'] == cat]['name'].apply(lambda x:  x if slim_dict.get(x) else None)
                    df_ranking.loc[
                        df_ranking['rank_id'] == cat,
                        'name'
                    ] = df_ranking[df_ranking['rank_id'] == cat]['name'].replace(slim_dict)
                else:
                    df_ranking.loc[
                        df_ranking['rank_id'] == cat,
                        'similarity'
                    ] = None

            df_ranking['similarity'] = df_ranking['similarity'].astype(float).fillna(1)

            df_all = pd.merge(df_athletes, df_ranking, on=['rank_id', 'name', 'country_code'])
            # new pdf in landscape

            pdf = PDF('L', tourname)

            cat_list_str = df_athletes['cat_name'].unique()

            for k in cat_list_str:
                pdf.add_page()
                pdf.alias_nb_pages()
                pdf.set_font("Arial", size=20)
                pdf.cell(200, 20, txt="Seeding for Category " + k, ln=1, align='C')

                st.header(k)

                if(len(df_athletes[df_athletes['cat_name'] == str(k)])<= 1):
                    st.info("Less than two athletes in category", icon="🚨")
                elif len(df_all[(df_all['cat_name'] == str(k))]) <= 0:
                    st.success("No one is in seeding", icon="ℹ️")
                else:
                    names_seeding = df_all[['name', 'country_code', 'ranking', 'totalpoints', 'similarity', 'original_name']][(df_all['cat_name'] == str(k))]
                    names_seeding['ranking'] = names_seeding['ranking'].astype(int)
                    names_seeding = names_seeding.sort_values(by=['ranking'], ascending=True)
                    names_seeding['position'] = list(range(1, len(names_seeding.index)+1))

                    # move positions to first column
                    cols = names_seeding.columns.tolist()
                    cols = cols[-1:] + cols[:-1]
                    names_seeding = names_seeding[cols]

                    # remove more than 4 seeded people
                    names_seeding = names_seeding[names_seeding['position'] < 5]
                    names_seeding = names_seeding.astype(str)

                    if len(names_seeding) > 0:
                        if names_seeding[names_seeding["similarity"].astype(float) < 1.0].empty:
                            st.write(names_seeding[['name', 'country_code', 'ranking', 'totalpoints']])
                        else:
                            st.warning('There are non exact matches, check names and original_name', icon="⚠️")
                            names_seeding["similarity"] = names_seeding["similarity"].astype(float).round(2)
                            st.dataframe(names_seeding.style.highlight_between(subset=['similarity'], left=0.1, right=0.99, color="#F31C2B"))

                            names_seeding = names_seeding.astype(str)
                            pdf.cell(200, 20, txt='!!! There are non exact matches, check names in event and ranking', ln=1, align='C')

                        if len(names_seeding['totalpoints'].unique())<len(names_seeding['totalpoints']):
                            st.error('Athletes have the same number of points, please make a pre-draw', icon="🚨")
                            pdf.cell(200, 20, txt='!!! Athletes have the same number of points, please make a pre-draw', ln=1, align='C')

                        fig = draw_as_table(names_seeding)
                        PNG_NAME = str(k) + ".png"
                        fig.write_image(PNG_NAME)
                        pdf.image(PNG_NAME)
                        os.remove(PNG_NAME)
                    else:
                        st.write("No one in Seeding")
                        pdf.set_font("Arial", size=15)
                        pdf.cell(200, 20, txt="No one in Seeding", ln=1, align='L')

            pdf.output("dummy2.pdf")
            with open("dummy2.pdf", "rb") as pdf_file:
                PDFbyte2 = pdf_file.read()

            fname = tourname + ' Seeding.pdf'
            st.download_button(label="Download Seeding",
                               data=PDFbyte2,
                               file_name=(fname))
            os.remove("dummy2.pdf")

        else:
            st.error("The event has no ranking categories, use a different event")

    else:
        st.error("The event has no categories, use a different event")

st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
