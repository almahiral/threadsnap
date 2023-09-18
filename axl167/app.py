import streamlit as st
from streamlit import session_state
from streamlit_extras.stateful_button import button
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Model, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, EncoderDecoderModel
import torch
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from get_twt_thread import get_thread, get_text
from indoNLP.preprocessing import remove_stopwords
from preprocess import clean_until_normalize, clean_until_concat, slang_search, preprocessing
import os
import pickle
import tensorflow as ts
from wordcloud import WordCloud
from rouge_score import rouge_scorer

st.title('ThreadSnap')

def get_twitter_thread(url):
    all = get_thread(url)
    all_text = get_text(all)
    return all_text

@st.cache_resource
def load_model():
    # model_name = "cahya/bert2bert-indonesian-summarization"
    model_name = "almahiral/mt5-base-indonesian-summarization"
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    # tokenizer.bos_token = tokenizer.cls_token
    # tokenizer.eos_token = tokenizer.sep_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner('Loading model...'):
    tokenizer, model = load_model()

def get_summary(processed_thread):
    input_ids = tokenizer.encode(processed_thread, return_tensors='pt')
    summary_ids = model.generate(
        input_ids,
        min_length=100,
        max_length=200,
        repetition_penalty=2.5,
        length_penalty=1,
        # early_stopping=True,
        no_repeat_ngram_size=3,
        # use_cache=True,
        # do_sample=True,
        # temperature = 0.8,
        top_k = 40,
        top_p = 0.75,
        num_beams=5
        )

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def get_rouge_score(processed_thread, summary):

    scorer = rouge_scorer.RougeScorer(['rougeLsum'])
    scores = scorer.score(processed_thread, summary)
    return scores

def create_wordcloud(text):
    wordcloud = WordCloud(width = 1000, height = 600, max_words = 100,
                        random_state=1, background_color="black", colormap="pink",
                        collocations=False).generate(text)
    return wordcloud

def word_frequency(normalized):
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    new_tokens = remove_stopwords(normalized)
    new_tokens = word_tokenize(new_tokens)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    # counts the word
    counted = Counter(new_tokens)
    # make dataframe
    word_freq = pd.DataFrame(counted.items(),columns=
                             ['word', 'frequency']).sort_values(by='frequency',ascending=False)
    return word_freq

# initialize state
# if 'stage' not in st.session_state:
#     st.session_state.stage = 0

# def set_stage(stage):
#     st.session_state.stage = stage


# if "summarize" not in st.session_state:
#     st.session_state["summarize"] = False

# if "visualize" not in st.session_state:
#     st.session_state["visualize"] = False


menu = ["Twitter Thread", "Text"]
choice = st.sidebar.selectbox("Menu", menu)

# sidebar menu 1: Summarize
if choice == "Twitter Thread":
    st.header('Copy link, paste link, get summary!')
    st.subheader("Important:")
    st.text("1. Only works with Twitter thread in Indonesian language ")
    st.text("2. Only works with Twitter thread that contains 5 or more Tweets")
    st.text("3. Only works with Tweets in the past 30 days")

    url = st.text_input("Enter the starting Twitter URL thread here:")
    
    if st.button("Summarize"):
        # st.session_state["summarize"] = not st.session_state["summarize"]
        # print the thread
        st.subheader("Let's see the steps of each process!")
        st.subheader("First, get the thread from Twitter 🐥")
        with st.spinner("Fetching the thread..."):
            st.caption("↓ Hit the little blue arrow here to minimize the thread.")
            thread = get_twitter_thread(url)
            st.write(thread)

        # print the cleaned thread
        st.subheader("Second, we combine and clean the thread ✨")
        st.caption("Remove URL, mention, emoji, emoticons, and more.")
        clean_concat = clean_until_concat(thread)
        st.write(clean_concat)

        # print the normalized thread
        st.subheader("Third, we replace slang with formal words ✍🏻")
        st.caption("Slangs and informal words are considered as noise! It could make the summarization bad.")
        normalized = clean_until_normalize(thread)
        st.write(normalized)
        st.markdown("**:red[Slangs replaced:]**")
        slang_found = slang_search(clean_concat)
        st.write(slang_found)

        # process the thread from clean to semantic search
        st.subheader("Fourth, remove unimportant sentence such as shameless self-promote, and others 👀")
        st.caption("Remove it so that it doesn't distract the summarizer.")
        processed_thread, sentence_removed = preprocessing(thread)
        st.write(processed_thread)
        st.markdown("**:red[Unimportant sentences removed:]**")
        st.write(sentence_removed)

        # print the summary
        st.subheader("Fifth and last step, we summarize the text! 📑")
        with st.spinner("Generating summary..."):
            st.caption("Enjoy speedy reading!")
            summary = get_summary(processed_thread)
            st.write(summary)

        # calculate ROUGE score
        with st.spinner("Generating accuracy score..."):
            st.caption("Evaluate how accurate the summary in numbers between 0 and 1. Higher value indicates better accuracy.")
            st.markdown("**Score 0: Precision    |    Score 1: Recall    |    Score 2: Fmeasure (harmonic mean of recall and precision)**")
            score = get_rouge_score(processed_thread, summary)
            st.write(score)
            st.caption("What do you think about the summary? Give us a feedback!")


        st.markdown("""---""")

        st.header("Your Thread, Visualized")
        st.text("See how you can see the thread in various visualizations! 📊")
        # button("Visualize", key="visualize")
        # if st.session_state["summarize"]:
            # if st.button("Visualize", on_click=set_stage, args=(2,)):
        with st.spinner("Loading graphs, charts, sprinkle of statistics..."):
            st.write("Word Cloud:")
            # wordcloud
            sw_removed = remove_stopwords(normalized)
            wordcloud = create_wordcloud(sw_removed)
            fig, ax = plt.subplots(figsize = (8,4))
            ax.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)
            
            # bar chart
            word_freq = word_frequency(normalized)
            fig2, axes = plt.subplots(figsize = (8,6))
            sns.barplot(x='frequency', y='word',data=word_freq.head(20))
            st.write('Most Common Words in Thread:')
            st.pyplot(fig2)
            # heatmap

# sidebar menu 2: enter text to summarize
if choice == "Text":
    st.header('Copy text, paste text, get summary!')
    st.subheader("Important:")
    st.text("Only works with text in Indonesian language.")

    text = [st.text_area("Enter text here:", height = 12)]
    
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            st.markdown("**:blue[Summary 1:]**")
            processed_text, sentence_removed = preprocessing(text)
            summary1 = get_summary(processed_text)
            print(processed_text)
            st.write(summary1)
            st.markdown("**:blue[Summary 2:]**")
            text1 = ' '.join(text)
            print(text1)
            summary2 = get_summary(text1)
            st.write(summary2)