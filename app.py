import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

emotions_info_dict = {
    "anger": "Anger is a strong feeling of displeasure or hostility.",
    "disgust": "Disgust is a feeling of revulsion or profound disapproval.",
    "fear": "Fear is an unpleasant emotion caused by the threat of danger, pain, or harm.",
    "happy": "Happiness is a state of well-being and contentment.",
    "joy": "Joy is a feeling of great pleasure and happiness.",
    "neutral": "Neutral is a state of no strong emotion.",
    "sad": "Sadness is a state of feeling sorrowful or unhappy.",
    "sadness": "Sadness is a state of feeling sorrowful or unhappy.",
    "shame": "Shame is a painful feeling of humiliation or distress caused by consciousness of wrong or foolish behavior.",
    "surprise": "Surprise is a feeling of mild astonishment or shock caused by something unexpected."
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Sentiment Analysis of Text using NLP")
    st.subheader("Detect the emotions in text")

    st.markdown("""
    ### What is Sentiment Analysis?
    Sentiment Analysis is the process of determining the emotional tone behind a series of words. It helps in understanding the sentiments expressed in texts.
    
    ### How does it work?
    Using Natural Language Processing (NLP) techniques, we analyze the text to predict the emotion behind it. NLP is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages.
    """)

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", placeholder="Enter text to analyze emotion...")
        submit_text = st.form_submit_button(label='Submit', type="primary")

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence: {:.2f}%".format(np.max(probability) * 100))
            st.info(emotions_info_dict[prediction])


        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', 
                y='probability', 
                color='emotions'
            ).properties(title="Prediction Probability Distribution")
            st.altair_chart(fig, use_container_width=True)

    st.sidebar.header("About")
    st.sidebar.info("This app uses NLP techniques to predict the emotion behind a given text. Developed by [Your Name].")
    st.sidebar.markdown("""
    ### Emotions Explained
    - **Anger**: 😠 A strong feeling of displeasure or hostility.
    - **Disgust**: 🤮 A feeling of revulsion or profound disapproval.
    - **Fear**: 😨😱 An unpleasant emotion caused by the threat of danger, pain, or harm.
    - **Happy**: 🤗 A state of well-being and contentment.
    - **Joy**: 😂 A feeling of great pleasure and happiness.
    - **Neutral**: 😐 A state of no strong emotion.
    - **Sad**: 😔 A state of feeling sorrowful or unhappy.
    - **Sadness**: 😔 A state of feeling sorrowful or unhappy.
    - **Shame**: 😳 A painful feeling of humiliation or distress caused by consciousness of wrong or foolish behavior.
    - **Surprise**: 😮 A feeling of mild astonishment or shock caused by something unexpected.
    """)

if __name__ == '__main__':
    main()
