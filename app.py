import pyperclip
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

if "bot_history" not in st.session_state:
    st.session_state["bot_history"] = []

if "user_history" not in st.session_state:
    st.session_state["user_history"] = []

st.title("Where Is My Data ? 🤖")

VECTOR_DB_PATH = "./vector_db"


@st.cache_resource
def dbt_to_vector_store(openai_api_key):
    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    return db


@st.cache_data
def build_input_to_llm(_final_docs, input_text):
    # TODO: Build prompt using dbt resource metadata (e.g., postgres provider)
    # TODO: How to handle with multiples users languages ? (e.g., pt-br, en-us, etc.)
    input = (
        "You will receive a series of information about data models"
        "from a generic company separated by <model>. Consider Postgres SQL as language"
        "available to run queries against these data models."
        "If you respond with snippets of SQL, put them in a specific format"
        "using ``` as separators from the rest of the response. For example for a model "
        "with name MODEL_NAME that belongs to the schema SCHEMA_NAME the query to select the model is:"
        "```sql"
        "select"
        "*"
        "from SCHEMA_NAME.MODEL_NAME"
    )
    for doc in _final_docs:
        if doc.metadata["resource_type"] == "model":
            input += "<model>"
            input += "SCHEMA_NAME: " + doc.metadata["schema"]
            input += "MODEL_NAME: " + doc.metadata["name"]
            input += (
                "Uma query que seleciona todas as linhas desse modelo é: "
                f"select * from {doc.metadata['schema']}.{doc.metadata['name']} "
            )
            input += "Descrição do modelo: " + doc.page_content
            input += "Colunas do modelo: " + doc.metadata["columns"]

            input += "<model>"
    input += (
        "Considerando os modelos definidos, me responda com o máximo de detalhes possível: "
        + input_text
    )
    return input


@st.cache_data
def generate_response(input_text, _db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    final_docs = _db.similarity_search(input_text, k=4)
    input_to_llm = build_input_to_llm(_final_docs=final_docs, input_text=input_text)
    llm_response = llm.call_as_llm(input_to_llm)
    return llm_response


def process_feedback(feedback):
    # TODO: Process the feedback here (e.g., store it in a database, update a model, etc.)
    st.write("Feedback received:", feedback)


@st.cache_data
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["schema"] = record.get("schema")
    metadata["name"] = record.get("name")
    metadata["resource_type"] = record.get("resource_type")
    metadata["columns"] = ""
    for column_name, column_data in record.get("columns").items():
        metadata["columns"] += f'{column_name}: {column_data["description"]}'
        metadata["columns"] += " "
    metadata["columns"] = str(metadata["columns"]).replace("\\", "")
    return metadata


@st.cache_resource
def load_data_to_docs(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".nodes[]",
        content_key="description",
        metadata_func=metadata_func,
    )
    return loader.load()


def initialize_vector_db(dbt_project_path, openai_api_key):
    all_docs = load_data_to_docs(dbt_project_path + "/target/manifest.json")
    vectordb = Chroma.from_documents(
        all_docs,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory=VECTOR_DB_PATH,
    )
    vectordb.persist()
    db = dbt_to_vector_store(openai_api_key)
    return db


api_key = st.sidebar.text_input("OpenAI API Key", type="password")
dbt_project_path = st.sidebar.text_input("Dbt project path", value="./data_models")

with st.form(key="input_form"):
    st.write("Welcome to *Where Is My Data ?*")
    st.write("Ask me anything about your data.")
    user_input = st.text_input(
        value="What is the name of the tables that contains the customer data ?",
        label="Enter some text",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not api_key:
            st.error("Please, provide an OpenAI API Key")
            st.stop()
        if not dbt_project_path:
            st.error("Please, select a dbt path first")
            st.stop()
        db = initialize_vector_db(dbt_project_path, api_key)
        response = generate_response(user_input, db)
        st.session_state.user_history.append(user_input)
        st.session_state.bot_history.append(response)

with st.form(key="output_form"):
    if st.session_state.bot_history:
        st.write(st.session_state.bot_history[-1])

        left, center, right = st.columns(spec=[10, 1, 1])

        copy_button = left.form_submit_button("📋")
        like_feedback = center.form_submit_button("👍")
        unlike_feedback = right.form_submit_button("👎")

        if like_feedback:
            process_feedback(" Like 🙂")

        if unlike_feedback:
            process_feedback(" Unlike 😞")

        if copy_button:
            pyperclip.copy(st.session_state.bot_history[-1])
            st.success("Copied to clipboard")
