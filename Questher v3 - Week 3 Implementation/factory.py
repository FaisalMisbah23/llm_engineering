from models import ModelManager

_mm = ModelManager()


class ProviderFactory:
    @staticmethod
    def create(provider: str):
        return _mm.get_client(provider)