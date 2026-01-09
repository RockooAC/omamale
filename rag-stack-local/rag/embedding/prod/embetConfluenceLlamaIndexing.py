#!/usr/bin/python3


import requests
import argparse
import langdetect
from typing import NoReturn, Dict, List
from ingestionPipeline import ZosIngestion
from llama_index.core.schema import Document


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection to embed data into"
    )
    args.add_argument(
        "--cache_collection",
        type=str,
        required=False,
        default="cache",
        help="Redis database to cache data into"
    )
    args.add_argument(
        "--token",
        required=True,
        type=str,
        metavar="Confluence REST API Personal Access Token"
    )
    args.add_argument(
        "--team",
        required=False,
        type=str,
        choices=["ZOS", "ZRS"],
        default="ZOS",
        metavar="Redge Team name to download confluence"
    )
    return args.parse_args()


class ConfluenceEmbedder(ZosIngestion):
    def __init__(self):
        self.args = parser()
        self.documents = list()
        self.url = 'https://confluence.redge.com/rest/api/content/search'
        self.cql_query = self._team_mapping()[self.args.team]
        self.headers = {
            "Authorization": f"Bearer {self.args.token}",
            "Content-Type": "application/json",
        }
        self.params = {
            "cql": self.cql_query,
            "limit": 250
        }
        super().__init__(
            collection_name=self.args.collection,
            cache_collection_name=self.args.cache_collection
        )

    @staticmethod
    def _team_mapping():
        return {
            "ZOS": f'(space=ADZ AND type=page AND lastModified >= now("-1y")) OR (space=RMKMS AND type=page AND lastModified >= now("-1y"))',
            "ZRS": f'(space=NJJ AND type=page AND lastModified >= now("-1y")) OR (space=IIS AND (title="NJJ" OR ancestor = 155743323) AND lastModified >= now("-1y"))',
        }

    def main(self) -> NoReturn:
        self.logger.info("Running Confluence embedding ingestion pipeline")
        self.pipeline = self.__ingestion_pipeline__(
            transformations=[
                self.__doc_splitter__,
                self.__embedder__,
            ],
            vector_store=self.__bm42_hybrid_vector_store__,
            docstore=self.__docstore__,
            docstore_strategy=self.insert_strategy,
            cache=self.__cache__,
        )
        # self._process_confluence_pages()

        nodes = self.pipeline.run(
            show_progress=True,
            documents=self.documents
        )

        self.logger.info(f"Ingested {len(nodes)} nodes")


    def _process_confluence_pages(self) -> NoReturn:
        self.logger.info("Running PL language Confluence pages removal")
        for doc in self._read_confluence_pages():
            chunks = [
                Document(text=chunk, metadata=doc.metadata)
                for chunk in self.__doc_splitter__.split_text(doc.get_content())
            ]
            try:
                if langdetect.detect(doc.get_content()) != "pl":
                    self.documents.extend(chunks)
            except langdetect.LangDetectException:
                self.documents.extend(chunks)

    def _read_confluence_pages(self) -> ZosIngestion.__confluence_reader__:
        self.logger.info("Running confluence pages reading")
        return self.__confluence_reader__(
            access_token=self.args.token
        ).load_data(
            include_attachments=False,
            page_ids=self._get_confluence_pages_ids_from_last_year(),
        )

    def _get_confluence_pages_ids_from_last_year(self) -> List[str]:
        self.logger.info("Running last year modified Confluence pages processing")
        return [str(page["id"]) for page in self._query_confluence_pages()["results"]]

    def _query_confluence_pages(self) -> Dict:
        self.logger.info("Querying Confluence for last year added pages")
        return requests.get(
            url=self.url,
            headers=self.headers,
            params=self.params
        ).json()


if __name__ == "__main__":
    ConfluenceEmbedder().main()
