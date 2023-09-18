state = {}

if st.button("Summarize"):
    # perform summarization and store the results in the state dictionary
    state["thread"] = get_twitter_thread(url)
    state["clean_concat"] = clean_until_concat(state["thread"])
    state["normalized"] = clean_until_normalize(state["thread"])
    state["processed_thread"] = preprocessing(state["thread"])
    state["summary"] = get_summary(state["processed_thread"])

    # show a message indicating that the results are ready
    st.success("Summarization complete.")

# show the outputs if they are present in the state dictionary
if "thread" in state:
    st.subheader("Thread from Twitter")
    st.write(state["thread"])

if "clean_concat" in state:
    st.subheader("Cleaned Thread")
    st.write(state["clean_concat"])

if "normalized" in state:
    st.subheader("Normalized Thread")
    st.write(state["normalized"])

if "processed_thread" in state:
    st.subheader("Processed Thread")
    st.write(state["processed_thread"])

if "summary" in state:
    st.subheader("Summary")
    st.write(state["summary"])

# show the visualize button if the summary is present in the state dictionary
if "summary" in state:
    if st.button("Visualize"):
        # visualize the results
        st.write("Words contained in the thread:")
        st.write("Slangs replaced:")
        st.write("Emoji removed:")
