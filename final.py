import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm



#st.session_state["warned_about_save_answers"] = True
st.set_page_config(layout="wide", page_title="Chest X-ray Report Generator", page_icon="")

@st.cache(allow_output_mutation=True)
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer

# # hide_streamlit_style = """
# # <style>
# # #MainMenu {visibility: hidden;}
# # footer {visibility: hidden;}
# # </style>
# # """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    st.title("Chest X-ray Report Generator")
    
    with st.container():
        paragraph="""\nThis app will generate impression part of an X-ray report.
                \nYou can upload 2 X-rays that are front view and side view of chest of the same individual.\nNote: The 2nd X-ray is optional."""
        
        text_area = st.empty()
        text=text_area.text_area("", paragraph, height=230)
    choices=['uploaded_files','sample_data']
    choice=st.selectbox("Select Activity to Predict on: ",choices)


test_data=None
if choice =="uploaded_files":
    col1,col2 = st.columns(2)
    col1.subheader("X-ray 1")

    image_1 = col1.file_uploader("X1",type=['png','jpg','jpeg'])
    image_2 = None
    col2.subheader("X-ray 2 (optional)")
    image_2 = col2.file_uploader("X2",type=['png','jpg','jpeg'])

    col1,col2 = st.columns(2)
    predict_button = col1.button('Predict on uploaded files')

elif choice =="sample_data":
    test_data=True

def predict(image_1,image_2,model_tokenizer,predict_button = None):
    caption=None
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
            image_1 = np.array(image_1)/255
            if image_2 is None:
                image_2 = image_1
            else:
                image_2 = Image.open(image_2).convert("RGB") #converting to 3 channels
                image_2 = np.array(image_2)/255
            st.image([image_1,image_2],width=300)
            caption = cm.function1([image_1],[image_2],model_tokenizer)
            if caption:
                text_area=st.empty()
                text = text_area.text_area("","Generated Rebort:\n"+ caption[0])

            time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            st.subheader(time_taken)

            del image_1,image_2
        else:
            st.markdown("## Upload an Image")

def predict_sample(model_tokenizer,folder = './test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1,no_files)
    file_path = os.path.join(folder,str(file))
    if len(os.listdir(file_path))==2:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = os.path.join(file_path,os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1,image_2,model_tokenizer,True)





model_tokenizer = create_model()

if test_data:
    predict_sample(model_tokenizer)
else:
    predict(image_1,image_2,model_tokenizer,predict_button=predict_button)





