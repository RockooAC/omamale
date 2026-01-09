"""
title: Basic langchain agent
description: A pipeline to integrate WebUI with langchain framework
requirements: langchain, langchain-community
"""

from typing import List, Union, Generator, Iterator
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


class Pipeline:
    def __init__(self):
        self.llm = Ollama(model="llama3:70b", base_url="http://10.255.240.156:11434")
        self.prompt = None

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

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(messages)
        print(user_message)

        chain = self.prompt | self.llm

        for chunk in chain.stream({"chat_history": messages, "context": "", "query": user_message}):
            yield chunk
