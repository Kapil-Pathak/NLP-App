import streamlit as st
import numpy as np
import pickle
from answer import answergen, answergen_bert, answergen_albert, answergen_xlm
from spacysum import summarization_spacy
from t5sum import question_generation, summary_t5, beam_search_decoding, paraphraser


def main():
  """ NLP ToolBox """
  st.image('download.png', width=350)
  # Title
  st.title("NLP ToolBox Streamlit")
  st.subheader("Natural Language Processing App for Beginners")
  st.subheader("""
      What are you interested in?
      """)

  # AnswerGeneration

  if st.checkbox("Answer Extraction"):
    st.subheader("Extract Answer from text")

    context =  st.text_area("Type your context here")
    question= st.text_input('Question: ', None)
    st.subheader("Which model you want to try?")
    if st.button("BERT"):
        answer = answergen_bert(context, question)
        st.success(answer)
    if st.button("DistilBERT"):
        answer = answergen(context, question)
        st.success(answer)
    

# Summarization
  if st.checkbox("Show Text Summarization"):
    st.subheader("Summarize Your Text")

    message = st.text_area("Enter Text","Type Here ..")
    summary_options = st.selectbox("Choose Summarizer",['spacy','T-5'])
    if st.button("Summarize"):
      if summary_options == 'spacy':

        # st.warning("Using Default Summarizer")
        st.text("Using Spacy ..")
        summary_result = summarization_spacy(message)
      else:
        st.text("Using T5")
        summary_result = summary_t5(message)


      st.success(summary_result)

  # Paraphrasing
  if st.checkbox("Show Question Paraphrasing"):
    st.subheader("Create similar questions")
    question= st.text_input('Question: ', None)

    if st.button("Paraphrase"):
      l = paraphraser(question)
      # st.success(l)
      st.write(l[0])
      st.write(l[1])
      st.write(l[2])
      # Paraphrasing
  if st.checkbox("Show Question Generation"):
    st.subheader("Create new question from context")
    message = st.text_area("Enter Text","Type Here ..")
    if st.button("Generate Questions"):
      l = question_generation(message)
      # st.success(l)
      st.write(l[0])
      st.write(l[1])
      st.write(l[2])


  st.sidebar.subheader("About ToolBox")
  st.sidebar.text("Natural Language Processing ToolBox")
  st.sidebar.info("Demo App")
  st.sidebar.text("By Kapil Pathak")

if __name__ == '__main__':
  main()
