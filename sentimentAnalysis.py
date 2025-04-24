import streamlit as st
import pandas as pd
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def analyze_sentiment(text):
    """Use GPT-4 to classify sentiment"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                {"role": "user", "content": str(text)}
            ],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("CSV Sentiment Analysis with GPT-4")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names and validate
        df.columns = df.columns.str.strip().str.lower()
        st.write("Detected columns:", df.columns.tolist())
        
        # Flexible column selection
        text_cols = [col for col in df.columns if 'feedback' in col]
        
        if not text_cols:
            st.error("No column containing 'Feedback' found in the CSV file")
            st.stop()
            
        text_col = text_cols[0]
        
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiments..."):
                # Analyze ALL rows (no head(50))
                df['gpt4_sentiment'] = df[text_col].apply(analyze_sentiment)
            
            st.success("Analysis complete!")
            st.write("Results:")
            st.dataframe(df[[text_col, 'gpt4_sentiment']])
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv,
                file_name='analyzed_sentiments.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
