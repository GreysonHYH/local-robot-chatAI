import { UnstructuredLoader } from "langchain/document_loaders/fs/unstructured" //load local files of many types
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter" //split the local file
import { PineconeClient } from "@pinecone-database/pinecone"  //vector database pinecone
import { OpenAIEmbeddings } from "langchain/embeddings/openai" //convert local data into vectors
import { PineconeStore } from "langchain/vectorstores/pinecone" //store local database into vector database 
import dotenv from 'dotenv'
dotenv.config()

//load local raw docs
const unstructuredLoader = new UnstructuredLoader("./vue3-document.md")
const rawDocs = await unstructuredLoader.load()

//split the local raw docs into chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:200
})
const docs = await splitter.splitDocuments(rawDocs)

//initialize vector library
const pineconeClient = new PineconeClient()
await pineconeClient.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
})
// construct an Index object
const pineconeIndex = pineconeClient.Index(process.env.PINECONE_INDEX)

//embed the markdown documents
try {
    PineconeStore.fromDocuments(docs,new OpenAIEmbeddings(),{
        pineconeIndex,
        textKey:"text",
        namespace:"vue3-document"
    })
} catch (error) {
    console.log(error)
}

