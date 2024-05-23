from src.datasets.services.tiny_shakespeare import TinyShakespeareProcessor
from src.datasets.services.wmt19_en_zh import WMT19EnZhProcessor

available_datasets = {
    "tiny_shakespeare": TinyShakespeareProcessor,
    "wmt19_en_zh": WMT19EnZhProcessor,
}
