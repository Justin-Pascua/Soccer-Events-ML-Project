import streamlit as st

with st.sidebar:
    st.markdown(""":material/attribution: :grey[Justin Pascua - 2025]  
                :material/work: :grey[[LinkedIn](https://www.linkedin.com/in/justin-pascua-673686187/)]  
                :material/code: :grey[[GitHub](https://github.com/Justin-Pascua)]""")

st.title(':material/school: Acknowledgements')
st.header("Raw match data")
st.write("""Citation: Pappalardo, Luca; Massucco, Emanuele (2019). Soccer match event dataset. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.4415000.v5""")
st.markdown("""Details: This dataset contains data on all events occuring in domestic 
            matches from Europe's top 5 leagues in the 2017/18 season, as well as matches 
            from the 2016 Euros and the 2018 World Cup. The linked page above provides a 
            *much more* detailed breakdown of the dataset. Much thanks to the dataset authors,
            Luca Pappalardo and Emanuele Massucco""")

st.header("Position label data")
st.markdown("""The above dataset was complimented by data from ESPN. I used the 
            `soccerdata` package to scrape ESPN's soccer data to obtain the exact 
            match-specific position label for each player. The package documentation can 
            be found here: https://soccerdata.readthedocs.io/en/latest/.""")

st.header("Inspiration")
st.markdown("""Lastly, I would be remiss to not mention my source of inspiration for this project. 
            It all started from watching this video, https://www.youtube.com/watch?v=zf8XzyBnqXk, 
            which discusses soccer from the perspective of network theory using the dataset mentioned 
            at the top of this page. Having taken a class in network theory, and being a lifelong 
            soccer fan, this video compelled me to explore the dataset for myself. Doing so, I eventually
            developed this project, which you're interacting with now. So, thank you to YouTuber "Not David" for putting together a well-crafted video.
            """)