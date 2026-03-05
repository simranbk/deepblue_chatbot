from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

# Your Supabase details
SUPABASE_BASE_URL = "https://yincbugwwaqiftbhomfu.supabase.co/storage/v1/object/public/isl-videos"
# https://yincbugwwaqiftbhomfu.supabase.co/storage/v1/object/public/isl-videos/alive.MOV

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)

#csv with label->vid_name mapping
df = pd.read_csv("label_mapping.csv")

documents = []
for index, row in df.iterrows():
    label = row['label']
    filename = row['filename']
    
    # construct the public streaming URL
    video_url = f"{SUPABASE_BASE_URL}/{filename}"
    
    # create the document. The label is embedded; the URL is metadata   
    doc = Document(
        page_content=label,
        metadata={"video_url": video_url}
    )
    documents.append(doc)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Successfully embedded {len(documents)} labels with Supabase URLs!")