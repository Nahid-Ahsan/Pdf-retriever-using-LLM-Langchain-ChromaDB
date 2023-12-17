from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_settings

check_point = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(check_point)
base_model = AutoModelForSeq2SeqLM.from_pretrained(check_point, 
                                            device_map = "auto", 
                                            torch_dtype = torch.float32
                                            )


def llm_pipline():
    pipe = pipeline(
        "text2text-generation",
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        # top_n = 0.95
    )

    local_llm = HuggingFacePipeline(pipeline = pipe)

    return local_llm

def qa_llm():
    llm = llm_pipline()
    embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    db = Chroma(persist_directory = "db", embedding_function = embeddings, client_settings = CHROMA_settings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        # chain_type="map_reduce",
        chain_type = "stuff",
        retriever = retriever,
        # return_source_document = True,
    )
    return qa


def process_answers(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    print(answer)
    return answer, generated_text

query = "is here any OCR system"

if __name__ == "__main__":
    process_answers(query)
    