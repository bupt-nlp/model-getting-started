from __future__ import annotations
import pytest

from transformers import BertConfig, BertTokenizer

@pytest.mark.webtest
class TestTransformerBert:
    def setup(self):
        model_name = ''
        self.config = BertConfig.from_pretrained()  
    def test_startup(self):
        pass

    def test_startup_and_more(self):
        pass