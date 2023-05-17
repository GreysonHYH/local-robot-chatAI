import { UnstructuredLoader } from "langchain/document_loaders/fs/unstructured" //read file
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { PineconeClient } from "@pinecone-database/pinecone"  //向量数据库
import { PineconeStore } from "langchain/vectorstores/pinecone" //将本地数据存入向量库 
import { OpenAIEmbeddings } from "langchain/embeddings/openai" //将数据转换为向量的过程
import dotenv from 'dotenv'

dotenv.config() //get .env and import to pineconeClient.init

const unstructuredLoader = new UnstructuredLoader("./vue3-document.md")

const rawDocs = await unstructuredLoader.load()

console.log(rawDocs)

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:200
})

const docs = await splitter.splitDocuments(rawDocs)

console.log(docs)

const pineconeClient = new PineconeClient()
await pineconeClient.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
})

const pineconeIndex = pineconeClient.Index(process.env.PINECONE_INDEX)

try {
    PineconeStore.fromDocuments(docs,new OpenAIEmbeddings(),{
        pineconeIndex,
        textKey:"text",
        namespace:"vue3-document"
    })
} catch (error) {
    console.log(error)
}

