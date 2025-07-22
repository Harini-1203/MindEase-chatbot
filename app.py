import streamlit as st
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json

# Load resources and initialize
nltk.download('popular')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

therapists = [
    {"name": "Dr. Aisha Sharma", "specialization": "Anxiety, Depression", "location": "Delhi", "contact": "aisha.sharma@example.com"},
    {"name": "Dr. Rohan Mehta", "specialization": "Stress Management, PTSD", "location": "Mumbai", "contact": "rohan.mehta@example.com"},
    {"name": "Dr. Kavita Rao", "specialization": "Child Therapy, Family Counseling", "location": "Bangalore", "contact": "kavita.rao@example.com"},
    {"name": "Dr. Arjun Kapoor", "specialization": "Addiction Recovery, Stress Management", "location": "Chennai", "contact": "arjun.kapoor@example.com"},
    {"name": "Dr. Priya Iyer", "specialization": "Couples Therapy, Relationship Counseling", "location": "Hyderabad", "contact": "priya.iyer@example.com"},
    {"name": "Dr. Sameer Gupta", "specialization": "Depression, General Counseling", "location": "Pune", "contact": "sameer.gupta@example.com"},
    {"name": "Dr. Nisha Desai", "specialization": "Family Counseling, Grief Support", "location": "Ahmedabad", "contact": "nisha.desai@example.com"},
    {"name": "Dr. Varun Malhotra", "specialization": "Workplace Stress, Anger Management", "location": "Kolkata", "contact": "varun.malhotra@example.com"},
    {"name": "Dr. Sneha Kulkarni", "specialization": "Teen Counseling, Anxiety", "location": "Pune", "contact": "sneha.kulkarni@example.com"},
    {"name": "Dr. Rajesh Nair", "specialization": "Mental Wellness, PTSD", "location": "Kochi", "contact": "rajesh.nair@example.com"}
]

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if ints:
        tag = ints[0]['intent']
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    ints = predict_class(msg)
    return get_response(ints)

# Streamlit app
def displayChat():
        # User input form
    st.header("Chat with the Mental Health Support Bot")
    chat_placeholder = st.container()
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your Question:", placeholder="Ask about your mental health")
        submit = st.form_submit_button("Send")

        if submit and user_input:
            response = chatbot_response(user_input)
            st.session_state['history'].append(("You", user_input))
            st.session_state['history'].append(("Chatbot", response))

    with chat_placeholder:
            if st.session_state['history']:
                for sender, message in st.session_state['history']:
                    st.markdown(f"**{sender}:** {message}")
                    st.markdown("<br>", unsafe_allow_html=True)
    st.write("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

def connect_to_therapists():
    st.header("Connect to Professional Therapists")
    specific_concern = st.selectbox(
        "Select your specific concern:",
        ["Anxiety", "Depression", "Stress Management", "PTSD", "Child Therapy", "Family Counseling"]
    )
    location = st.text_input("Enter your location (optional):", placeholder="e.g., Delhi")
    
    st.subheader("Available Therapists")
    matched_therapists = [
        therapist for therapist in therapists 
        if specific_concern in therapist["specialization"]
        and (not location or location.lower() in therapist["location"].lower())
    ]
    
    if matched_therapists:
        for therapist in matched_therapists:
            st.markdown(f"**Name:** {therapist['name']}")
            st.markdown(f"**Specialization:** {therapist['specialization']}")
            st.markdown(f"**Location:** {therapist['location']}")
            st.markdown(f"**Contact:** {therapist['contact']}")
            st.markdown("---")
    else:
        st.warning("No therapists found matching your criteria. Please try again with different preferences.")

def main():
    st.set_page_config(page_title="Mental Health Support Chatbot", page_icon="🧠", layout="wide")
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'greeted' not in st.session_state:
        st.session_state['greeted'] = False
    option = st.sidebar.radio("Choose an option:", ["Chatbot", "Connect to Therapists"])

    if option == "Chatbot":
        displayChat()
    elif option == "Connect to Therapists":
        connect_to_therapists()

    st.markdown("---")

if __name__ == "__main__":
    main()