from abc import ABC, abstractmethod
from typing import List


class QueryTransform(ABC):
    """
    Base interface for all query translation techniques.

    transform() takes the original user query and returns one or more
    query strings to retrieve with. The caller decides how to embed and
    retrieve each returned query.
    """

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """
        Args:
            query: The original user question.

        Returns:
            List of one or more query strings.
            - Single-item list: HyDE, step-back, simple rewrite
            - Multi-item list:  multi-query, decomposition
        """
        ...
