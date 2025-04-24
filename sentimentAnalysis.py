import streamlit as st
import pandas as pd
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_sentiment(model: str, text: str) -> str:
    """Get sentiment classification from specified model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                {"role": "user", "content": str(text)}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Multi-Model Sentiment Analysis")
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
            st.error("No column containing 'feedback' found in the CSV file")
            st.stop()
            
        text_col = text_cols[0]
        
        if st.button("Analyze Sentiment"):
            with st.spinner("Running sentiment analysis across models..."):
                # Create new columns for each model
                models = [
                    ('gpt4_sentiment', 'gpt-4'),
                    ('gpt35_sentiment', 'gpt-3.5-turbo')
                ]
                
                for col_name, model_name in models:
                    df[col_name] = df[text_col].apply(
                        lambda x: get_sentiment(model_name, x)
                    )
            
            st.success("Analysis complete!")
            st.write("Results:")
            st.dataframe(df[[text_col] + [m[0] for m in models]])  # Show all columns
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv,
                file_name='multi_model_sentiments.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
