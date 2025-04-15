import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime


load_dotenv()


st.title("LexiForms ⚖️")


if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None


data = []  


# Add this function to generate a tenancy agreement
def generate_tenancy_agreement(landlord_name, tenant_name, landlord_address, tenant_address, property_address, duration, start_date, monthly_rent, amount_in_words, deposit_amount, deposit_in_words, day_of_month, notice_period, witness_landlord, witness_tenant, witness_address_landlord, witness_address_tenant):
    # Get current date information
    today = datetime.now()
    current_day = today.day
    current_month = today.strftime("%B")  # Full month name
    current_year = today.year
    
    # Create ordinal suffix for day (1st, 2nd, 3rd, etc.)
    day_suffix = "th" if 11 <= current_day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(current_day % 10, "th")
    formatted_day = f"{current_day}{day_suffix}"
    
    return f"""
    THIS TENANCY AGREEMENT is made on the {formatted_day} day of {current_month}, {current_year}.
    BETWEEN:
    {landlord_name}, of {landlord_address} (hereinafter referred to as "the Landlord")
    AND
    {tenant_name}, of {tenant_address} (hereinafter referred to as "the Tenant")
    IT IS AGREED AS FOLLOWS:
    1. The Property
    The Landlord lets and the Tenant takes ALL THAT property situated at {property_address} (the "Property").

    2. Term
    The tenancy shall be for a period of {duration}, commencing on {start_date}.

    3. Rent
    Monthly rent is {monthly_rent} ({amount_in_words}) Ghana Cedis, payable in advance on the {day_of_month} of each month.

    4. Security Deposit
    The Tenant shall pay a deposit of {deposit_amount} ({deposit_in_words}) Ghana Cedis, refundable at the end of the tenancy, subject to deductions.

    5. Tenant's Covenants
    The Tenant shall:
    - Pay rent on time
    - Maintain the property in good condition
    - Avoid damage or illegal use
    - Permit reasonable inspections
    - Not sublet without consent

    6. Landlord's Covenants
    The Landlord shall:
    - Guarantee quiet possession
    - Maintain the property's structure

    7. Termination
    Either party may terminate the agreement with {notice_period} written notice.

    8. Governing Law
    This Agreement shall be governed by the laws of the Republic of Ghana.

    SIGNED:
    Landlord: _____________________
    Name: {landlord_name}
    Tenant: _______________________
    Name: {tenant_name}
    Witnesses:
    Name: {witness_landlord}, Address: {witness_address_landlord}
    Name: {witness_tenant}, Address: {witness_address_tenant}
    """


# Modify the sidebar to include an option for generating a tenancy agreement
with st.sidebar:
    st.title("Menu:")
    option = st.radio("Choose an option:", ["Upload a PDF", "Generate Tenancy Agreement"])

    if option == "Upload a PDF":
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        if st.button("Submit & Process") and uploaded_file:
            with st.spinner("Processing..."):
                temp_file_path = f"./temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_file_path)
                data = loader.load()
                os.remove(temp_file_path)
    elif option == "Generate Tenancy Agreement":
        landlord_name = st.text_input("Landlord's Name")
        landlord_address = st.text_input("Landlord's Address")
        tenant_name = st.text_input("Tenant's Name")
        tenant_address = st.text_input("Tenant's Address")
        property_address = st.text_input("Property Address")
        duration = st.text_input("Duration (e.g., 2 years)")
        start_date = st.text_input("Start Date (e.g., 1st January 2025)")
        monthly_rent = st.text_input("Monthly Rent (e.g., GHS 500)")
        amount_in_words = st.text_input("Rent Amount in Words (e.g., Five Hundred Ghana Cedis)")
        deposit_amount = st.text_input("Deposit Amount (e.g., GHS 1000)")
        deposit_in_words = st.text_input("Deposit Amount in Words (e.g., One Thousand Ghana Cedis)")
        day_of_month = st.text_input("Day of Month Rent is Due (e.g., 1st)")
        notice_period = st.text_input("Notice Period (e.g., 1 month)")
        witness_landlord = st.text_input("Witness for Landlord")
        witness_address_landlord = st.text_input("Witness Address for Landlord")
        witness_tenant = st.text_input("Witness for Tenant")
        witness_address_tenant = st.text_input("Witness Address for Tenant")

        if st.button("Generate Agreement"):
            agreement = generate_tenancy_agreement(
                landlord_name, tenant_name, landlord_address, tenant_address, property_address,
                duration, start_date, monthly_rent, amount_in_words, deposit_amount, deposit_in_words,
                day_of_month, notice_period, witness_landlord, witness_tenant,
                witness_address_landlord, witness_address_tenant
            )
            st.text_area("Generated Tenancy Agreement:", agreement, height=500)


if data:
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="./chroma_db"
    )
    
    
    st.session_state.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    st.success("Done")


query = st.chat_input("Say something: ")

if query and st.session_state.retriever:  
    st.session_state.history.append({"user": query})

    
    system_prompt = (
        "You are a specialized legal assistant providing information based on legal documents. "
        "Use the following retrieved context to answer questions about legal matters and documents. "
        "Prioritize accuracy and clarity in your explanations. "
        "When interpreting legal language, be precise but accessible to non-specialists. "
        "If the answer is not clearly found in the context or requires legal advice beyond document interpretation, "
        "clarify that you can only provide information based on the document and not legal advice. "
        "If a question falls outside the scope of the documents or requires professional legal judgment, "
        "recommend consulting with a qualified legal professional. "
        "Base your responses solely on the provided context, not on general legal knowledge. "
        "Format important terms, definitions, or clauses with appropriate emphasis when relevant. "
        "\n\n"
        "{context}"
    )

    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    
    question_answer_chain = create_stuff_documents_chain(
        ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0), 
        prompt
    )
    rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

    
    response = rag_chain.invoke({"input": query})

    
    st.session_state.history.append({"assistant": response['answer']})



if st.session_state.history:
    for chat in st.session_state.history:
        if "user" in chat:
            st.markdown(f"<div style='text-align: right; color: white;'><b>You:</b> {chat['user']}</div><br> ", unsafe_allow_html=True)
        if "assistant" in chat:
            st.markdown(f"<div style='text-align: left; color: white;'><b>Assistant:</b> {chat['assistant']}</div><br> ", unsafe_allow_html=True)