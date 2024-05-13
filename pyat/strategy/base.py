from abc import ABC, abstractmethod
from typing import Dict


class BaseStrategy(ABC):
    @abstractmethod
    def select_item(self, session: Dict) -> int:
        """

        :param session: {
            'unselected': set(item_no)
            'selected': set(item_no)
        }
        :return: item_no
        """
        pass
