import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import json

st.set_page_config(
#     layout = 'wide'
)
st.title(":material/lightbulb: Methodology")
st.markdown(":material/construction: UNDER CONSTRUCTION :material/construction:")
with st.sidebar:
    st.markdown(""":material/attribution: :grey[Justin Pascua - 2025]  
                :material/work: :grey[[LinkedIn](https://www.linkedin.com/in/justin-pascua-673686187/)]  
                :material/code: :grey[[GitHub](https://github.com/Justin-Pascua)]""")

st.header("Overview")
st.write("""To visualize a match, we take raw match data (see the "Acknowledgments" 
            page) in order to determine the number of passes between each player, each 
            player's position (e.g. GK, CB, LW, etc.), and each team's formation. We do 
            this using a combination of machine learning models, and non-ML algorithms. 
            In the following sections, we give a brief overview of the datasets used, 
            how player positions and team formations are predicted, and how the passing 
            networks are formed.""")

tab_names = ["The Data", "Player Position Prediction", 
             "Formation Inference", "Passing Networks"]
data_tab, ml_model_tab, formation_inf_tab, passing_networks_tab = st.tabs(tab_names)
with data_tab:
    st.header(tab_names[0])
    st.write("""This project drew from two sources of data, which we'll refer to as 
             i) the Wyscout dataset, and ii) the ESPN dataset. For more details, refer 
             to the "Acknowledgments" page.""")
    
    st.subheader("The Wyscout Dataset")
    st.write("""The Wyscout dataset contains data on domestic matches for Europe's 
             top 5 leagues, as well as data on the 2016 European Championship and the 
             2018 World Cup. Raw match data is stored as a table of "events" (e.g. 
             passes, shots, fouls, set pieces, etc.) that occurred in a given match. 
             For example, the first couple of rows of the table for the 2018 France vs. 
             Argentina match looks as follows:""")
    
    sample_df = pd.read_csv('static/FranceArgentinaWCEventsDf.csv')
    st.dataframe(sample_df)
    st.caption("First 5 rows of the raw match data for the France vs Argentina, World Cup 2018 match")
    
    st.write("""Each event instance comes with additional information indicating which 
             player committed the event, where on the pitch the event occurred, and 
             whether or not the event was accurate (if applicable). (Note: the original 
             dataset contains more fields, but for our purposes, this information is all 
             we care about.""")
    st.write("""Per the Wyscout API documentation, the coordinates in the dataset are of the 
            form $(x, y)$, where $x$ and $y$ range over integers in the interval $[0, 100]$. 
             The pitch is oriented as follows:""")
    
    cols = st.columns([1, 5, 1])
    with cols[0]:
        st.write('')
    with cols[1]:
        st.image('static/WyscoutDataCoordinates.png')
        st.caption('Wyscout coordinate conventions. Source: https://apidocs.wyscout.com/#section/Data-glossary-and-definitions/Pitch-coordinates')
    with cols[2]:
        st.write('')
    st.write("""Here, the frame of reference depends on which team the player is on, 
             so that the player's own goal is always on the line $x = 0$.""")
    
    st.subheader("The ESPN Dataset")
    st.write("""The Wyscout dataset is rich in detail, but is in some sense incomplete. 
             In particular, it doesn't list team formations and player positions, both 
             of which are crucial for effectively understanding a match. (The dataset 
             does contain info about a player's typical role throughout the season, but 
             these are not match-specific, and are broad categories like "GK," "DF", 
             "MD", and "FW"). To address this, we complimented the Wyscout dataset with 
             data scraped from ESPN. (For more details about the scraping process, refer 
             to the "Acknowledgments" page.) The data obtained from ESPN does indeed 
             list match-specific player positions. ESPN has several detailed class labels,
             but I chose to merge similar positions (e.g. I treated LF as the same as LW, and LWB as LB). 
             This was helped address class imbalance without compromising the descriptiveness 
             needed to determine team formations. Ultimately, I settled on the following class labels:
             GK, LB, CB, RB, LM, CM, RM, LW, CF, and RW.""")
    st.write("""In order to match the ESPN dataset to the Wyscout dataset, I needed to 
             match ESPN player/team/game names to Wyscout player/team/game names. 
             Matching the team names was simple because there were only a handful of 
             teams whose names varied between the two datasets, and those were easy to 
             match up by eye. Once I matched team names, matching specific games between 
             the datasets was straightforward because both datasets identified matches 
             using a string of the form "TEAM 1 - TEAM 2, DATE" (or some variant). 
             Matching player names was much more difficult because both datasets used 
             somewhat inconsistent formats. The ESPN dataset records players' "popular" 
             full names, i.e. the full name that the general public might know a player 
             by. The Wyscout dataset records both the player's full legal name, and an 
             abbreviated version of their popular name. The problem with popular names, 
             I found, is that they differ between different sources (e.g. ESPN referred 
             to Hector Bellerin as "Hector Bellerin," but Wyscout referred to him as 
             "Bellerin"). I tried a couple tricks (e.g. using the full names provided by 
             Wyscout to constructive alternative popular names, and trying to match 
             Wyscout full names to ESPN popular names, etc.), but I still was only able 
             to match about 2/3 of player names in the Wyscout dataset to a player names 
             in the ESPN dataset.""")
    st.write("""After matching the two datasets, I then used the position labels in the 
             ESPN dataset to train an ML model to predict a player's position based on 
             where the player committed events throughout the match.""")

with ml_model_tab:
    st.header(tab_names[1])
    st.write("""In order to predict each player's position, we assume that their location 
             on the pitch throughout the game is a good indicator. However, our dataset 
             doesn't record the player's exact positions throughout the entire match. The 
             dataset only sees the player whenever they commit an event. So, we can only 
             base our predictions based on where the player committed events throughout the 
             match.""")
    st.write("""Given the raw match data, we can get all the coordinates for events 
             committed by each player. We do this by grouping the table based on the team 
             and player column, and the gathering coordinates. For the France vs Argentina 2018 
             match, this looks as follows:""")
    
    coords_df = pd.read_csv('static/FranceArgentinaWCCoordsDf.csv')
    st.dataframe(coords_df)
    st.caption("List of event coordinates of each player in the France vs Argentina, World Cup 2018 match")
    
    st.write("""Now, to restructure this data in a form suitable for a machine learning 
             model, we use each list of coordinates to create a sparse 2D array. This is 
             done exactly how one might expect. We start with a $(100, 100)$-shaped array of 
             zeroes. Then, for each coordinate in the coordinate list, we add a one
             to the array at the position specified by the coordinate. For example, if
             we do this for Kylian Mbappe's coordinates from the table above, we have:""")
    
    fig1 = pio.read_json('static/SparseFig.json')
    st.plotly_chart(fig1)
    st.caption("Map of Kylian Mbappe's events in the France vs Argentina, World Cup 2018 match")
    
    st.write("""This gives us a sparse array of shape $(100, 100)$ showing us visually 
             where on the pitch the player committed events. In order to reduce the 
             number of parameters in our ML model, and in order to improve its 
             generalizability, we apply a Gaussian blur to the array, and resize it to 
             $(50, 50)$. This gives us images which look as follows:""")
    
    fig2 = pio.read_json('static/EventsMap.json')
    st.plotly_chart(fig2)
    st.caption("Kylian Mbappe's events map blurred and resized down to $(50, 50)$")

    st.write("""This preserves the general structure of the original array while reducing
             the actual size. We then predict the player's position using a deep neural 
             network. In particular, we flatten the events map (turning it into a 250-dim 
             vector) and feed it into an 4-layer MLP. This gives us a 10-dim probability 
             vector whose entries represent the probability the player is of the given 
             position. Feeding the above array into the model, we have the following 
             probabilities:""")    
    with open('static/MbappeProbs.json', 'r') as file:
        data = json.load(file)
    probs_df = pd.DataFrame(data = [data.keys(), 
                                    data.values()]
                            ).transpose()
    fig = px.bar(probs_df, x = 1, y  = 0, 
                    orientation = 'h', 
                    height = 250, 
                    width = 100,
                    color_discrete_sequence = ['#8A2BE2']
                    )
    fig.update_layout(
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        yaxis_title = 'Position',
        xaxis_title = 'Probability',
        margin = dict(l = 0, r = 0, t = 0, b = 0),
    )
    fig.update_xaxes(color = 'white')
    fig.update_yaxes(color = 'white')
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.plotly_chart(fig, width = 'stretch')
    st.caption("Position predictions for Kylian Mbappe in the France vs Argentina, World Cup 2018 match")
    st.write(f"""Note that we do not necessarily assign the player the position that has 
             the highest probability. For more details about how we actually determine 
             positions in the final match visualizer, refer to the {tab_names[2]} page.""")
    st.write("""To briefly explain, note we're predicting the player's position here solely
             based on the individual player's heatmap. However, the individual player's 
             heatmap should be understood in the context of the rest of the player's team.
             In other words, the way a player positions themselves depends on the positioning
             of the rest of the team. For example, a CF in a 4-3-3 formation may play more
             centrally because the wide areas of the pitch are occupied by the LW and RW players.
             On the other hand, a CF in a 5-3-2 may wander more wide because this formation has
             no wide attacking players.""")
    st.write("""This is all to say that, in order to properly assign player positions, one needs
             to consider the entire team rather than just the individual player.""")

with formation_inf_tab:
    st.header(tab_names[2])
    st.write("""As explained in the previous tab, in order to effectively determine a 
             team's formation, one should consider the positioning of the entire team. 
             To do so, we first apply the ML model described in the previous tab to each 
             of the 11 players on the starting 11. This gives us an $(11,10)$-shaped 
             array where the rows represent 11 players, and the columns represent the 
             10 possible positions. For the France vs Argentina, World Cup 2018 match, 
             this array for the French team looks as follows:""")
    fig = pio.read_json('static/FranceProbs.json')
    st.plotly_chart(fig)

with passing_networks_tab:
    st.header(tab_names[3])

# discuss...
# format of raw events_df 
# events_df -> heatmaps
# heatmaps -> [heatmap_model] -> predictions
# heatmaps -> [heatmap_autoencoder] -> heatmap_reconstruction

