Legal AI Chatbot with Tenancy Agreement Generator

This project is a Streamlit-based application that combines **retrieval-augmented generation (RAG) with Google Gemini AI to answer questions based on uploaded PDF documents and generate tenancy agreements. The app is designed to assist users in legal-related tasks, such as document analysis and agreement generation.



 Features

1. PDF Upload and Question Answering:
   - Upload a PDF document.
   - Ask questions about the content of the document.
   - The app retrieves relevant sections and provides concise answers.

2. Tenancy Agreement Generator:
   - Generate a tenancy agreement by filling out a form with details such as landlord/tenant names, property address, rent, and more.
   - The agreement is dynamically generated based on a predefined template.

3. Interactive Chat Interface:
   - Engage in a chat-like interface to ask questions and receive answers.



 Installation

 Prerequisites
- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)

 Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/legal-ai-chatbot.git
   cd legal-ai-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a .env file in the root directory.
   - Add your API keys for Google Gemini and other services:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open the app in your browser at `http://localhost:8501`.

---

 Usage

 Menu Options
- Upload a PDF:
  - Upload a PDF document.
  - Click "Submit & Process" to extract the content.
  - Use the chat input to ask questions about the document.

- Generate Tenancy Agreement:
  - Fill out the form with details such as:
    - Landlord's and Tenant's names and addresses.
    - Property address, rent, deposit, and duration.
    - Witness details.
  - Click "Generate Agreement" to create a tenancy agreement.
  - The generated agreement will be displayed in a text area.

---

 Example Queries for PDF Analysis
- "What is the main topic of this document?"
- "Summarize the key points in section X."
- "What does the document say about [specific topic]?"
- "Explain the methodology used in this paper."

---



 Technologies Used
- Streamlit: For building the web interface.
- LangChain: For document processing and retrieval-augmented generation.
- Google Gemini AI: For generating embeddings and answering questions.
- Chroma: For vector storage and similarity search.
- dotenv: For managing environment variables.

---

File Structure
```
.
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not included in the repo)
├── README.md              # Project documentation
└── other_files/           # Additional files (if any)
```

---

 Future Improvements
- Enhance the tenancy agreement generator with more customizable clauses.
- Integrate additional legal document templates.



 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

 Acknowledgments
- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Google Gemini AI](https://cloud.google.com/ai-platform)

Feel free to contribute or raise issues for improvements! 

