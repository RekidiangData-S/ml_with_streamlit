import time
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff

# set title
st.title("Streamlit App for Test")
st.subheader("Rekidiang DataS ")
image = Image.open('rekidiangdataslogo.jpg')
st.image(image, use_column_width=True)
st.write("Nous deployer le model avec streamlit ")
st.markdown("this markdown")
st.success("Felicitation")
st.info('information')
st.warning("be cautious")
st.error("Error : intall streamlit and rerun !!!!")
st.help(print)

df = np.random.rand(5, 15)
st.dataframe(df)

st.text('======'*25)

ds = pd.DataFrame(df, columns=('column %d' % i for i in range(15)))
st.dataframe(ds.style.highlight_max(axis=1))

st.text('======'*25)
# Display chart
chart = pd.DataFrame(np.random.rand(20, 3), columns=['a', 'b', 'c'])
st.line_chart(chart)
st.text('======'*25)
st.area_chart(chart)
st.text('===Bar Chart==='*25)
chart = pd.DataFrame(np.random.rand(50, 3), columns=['a', 'b', 'c'])
st.bar_chart(chart)
st.text('======'*25)
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=25)
st.pyplot()
st.text('======'*25)
x1 = np.random.randn(200)-2
x2 = np.random.randn(200)
x3 = np.random.randn(200)-2

hist_data = [x1, x2, x3]
groupe_labels = ['Goup1', 'group2', 'Group3']
fig = ff.create_distplot(hist_data, groupe_labels, bin_size=[.2, .25, .5])
st.plotly_chart(fig, use_container_width=True)

st.text('======'*25)
dfs = pd.DataFrame(np.random.randn(100, 2) /
                   [50, 50]+[37.76, -122.4], columns=['lat', 'lon'])
st.map(dfs)
st.text('===Press Buttons==='*25)
# Creating Buttons
if st.button("Say Hello"):
    st.write("hello is here")
else:
    st.write("why are you here")
st.text('===Radio==='*25)
# Radio
genre = st.radio("Your please ?", ('Male', 'Female'))
if genre == 'Male':
    st.write('You are strong')
elif genre == 'Female':
    st.write("You're Beautuful")
st.text('===Select button==='*25)
# Select button
option = st.selectbox("What is your Marrital status ?",
                      ('Single', 'Married', 'Devorced'))
st.write("You are ", option)
st.text('===Multiple Selection==='*25)
# Multiple Selection
option = st.multiselect("Which fruit do you prefer ?",
                        ('mango', 'apple', 'orange', 'banana', 'avocado'))
st.write("You prefer : ", option)

st.text('===Slider==='*25)
# Slider
age = st.slider("How old are you ?", 0, 120, 10)
st.write("You are : ", age, "year old")

value = st.slider("Select a range of values ?", 0, 150, (10, 50))
st.write("You Selected a range between : ", value)

st.text('===input number==='*25)
# input number
number = st.number_input('input number :')
st.write('The number you inputed is : ', number)

st.text('===upload files==='*25)
# upload files
upload_file = st.file_uploader('choose a csv file', type='csv')
if upload_file is not None:
    data = pd.read_csv(upload_file)
    st.write(data)
    st.success("Data Successfully Uploaded !")
else:
    st.info('please csv file')

# side bar
add_sidebarS = st.sidebar.selectbox(
    "What your favorite course ", ('Others', 'Not sure'))

st.text('===color picker==='*25)

# progress bar
progress_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    progress_bar.progress(percent_complete+1)

with st.spinner('wait a moment ....'):
    time.sleep(5)
st.success("it' Ready Now..")
# Balloon
st.balloons()
# color picker
color = st.beta_color_picker("pick your prefered color", '#00f900')
st.write("This your color code :", color)

# side bar
add_sidebarS = st.sidebar.selectbox(
    "What your favorite course ", ('Others', 'Not sure'))
