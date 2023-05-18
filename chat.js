import { ConversationalRetrievalQAChain } from "langchain/chains"
import { PineconeClient } from "@pinecone-database/pinecone"  //向量数据库
import { PineconeStore } from "langchain/vectorstores/pinecone" //将本地数据存入向量库 
import { OpenAIEmbeddings } from "langchain/embeddings/openai" //将数据转换为向量的过程
import { OpenAI } from "langchain/llms/openai"
import dotenv from 'dotenv'
dotenv.config()

const model = new OpenAI({
    temperature:0 // a => b 根据一个字符匹配下一个字符的准确度 0代表最准确的值 如果比较大就会随机找，对于创造性生成文章比较有趣
})

//创建向量数据库
const pineconeClient = new PineconeClient()
await pineconeClient.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
})
const pineconeIndex = pineconeClient.Index(process.env.PINECONE_INDEX)
const pineconeStore = await PineconeStore.fromExistingIndex(new OpenAIEmbeddings(),{
    pineconeIndex,
    textKey:"text",
    namespace:"vue3-document"
}) 

// 取向量数据库里面的数据
const chain = ConversationalRetrievalQAChain.fromLLM(model,pineconeStore.asRetriever(),{
    returnSourceDocuments:true  //返回参考来哪些原材料
})

// 提问题
const res = await chain.call({
    question:"我是vue新手，给我一些学习建议",
    chat_history:[]
})
//第二次提问题，携带历史记录
const secondRes = await chain.call({
    question:"是否可以提供更多的建议",
    chat_history:["我是vue新手，给我一些学习建议",res.text]
})

console.log(secondRes)