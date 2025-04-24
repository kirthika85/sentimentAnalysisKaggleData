import streamlit as st
import pandas as pd
import openai

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def analyze_sentiment(text):
    """Use GPT-4 to classify sentiment"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                {"role": "user", "content": text}
            ],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

# Streamlit UI
st.title("CSV Sentiment Analysis with GPT-4")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiments..."):
            # Analyze first 50 rows for demo (remove slicing for full analysis)
            df = df.head(50)
            df['GPT4_Sentiment'] = df['Text'].apply(analyze_sentiment)
        
        st.success("Analysis complete!")
        st.write("Results:")
        st.dataframe(df[['Text', 'GPT4_Sentiment']])
        
        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results",
            data=csv,
            file_name='analyzed_sentiments.csv',
            mime='text/csv'
        )
