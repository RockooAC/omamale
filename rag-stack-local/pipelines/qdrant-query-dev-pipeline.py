"""
title: Basic langchain agent
author: Damian Płaskowicki
date: 2024-07-26
version: 1.0
license: MIT
description: A pipeline to integrate WebUI with langchain framework
requirements: langchain, langchain-community, langchain-qdrant
"""

from typing import List, Union, Generator, Iterator
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import Qdrant


def qdrant_docs_parser(documents: list) -> str:
    output = ""
    for doc in documents:
        output += f"source: {doc.metadata['source']}\n```\n{doc.page_content}\n```\n"
    return output


class Pipeline:
    def __init__(self):
        self.llm = Ollama(model="llama3:70b", base_url="http://10.255.240.156:11434")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest", base_url="http://10.255.143.169:11434")
        self.prompt = None
        self.retriever = None

    async def on_startup(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                (
                    "system",
                    """I am the personal assistant to the system software team developing the products: Redge Media Coder and Redge Media Origin/CDN (packager). The team works on issues related to multimedia streaming (codecs, containers, protocols such as HLS, DASH, SS), as well as networking: UDP, TCP/IP, RTMP, MPEG-TS, multicast, and distribution to end-users. They also focus on aspects related to the performance of nodes (servers). Additionally, the team addresses video/audio codecs such as H.264, H.265, AAC, MPEG-2, and NVIDIA technologies including CUDA and NVENC.

            Your Motive:
            Your task is to analyze the provided (embedded) documents and deliver information based on context.

            Focus on Conciseness and Clarity:
            Ensure that the output is concise yet comprehensive. Focus on clarity and readability to provide researchers with easily digestible insights.

            Focus on providing comprehensive and very detailed answers. It is better to write more loosely related information than less strictly related information. Think high-level and connect the facts.

            Try also to add annotation to every information from ISO specification or document so user can verify it.

            IMPORTANT:
            If the user query cannot be answered using the provided context, do not improvise; you should only answer using the provided context from the research papers.
            Also when you meet conflicting information prefer the one from the newer version of the documentation/specification.
            If the user asks something that does not exist within the provided context, answer ONLY with: 'Sorry, the provided query is not clear enough for me to answer from the provided research papers'.

            CONTEXT
            {context}
            """,
                ),
                ("user", "{query}"),
            ]
        )

        store = Qdrant.from_existing_collection(
            embedding=self.embeddings,
            collection_name="standard_lg",
            url="http://10.255.146.104:6333",
        )
        self.retriever = store.as_retriever(search_kwargs={"k": 10})

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(messages)
        print(user_message)

        knowledgebase = self.prompt | self.retriever

        knowledge = knowledgebase.invoke({"chat_history": [], "context": "", "query": user_message})

        return qdrant_docs_parser(knowledge)
