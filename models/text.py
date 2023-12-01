from dataclasses import dataclass


@dataclass
class TextDoc:
    body: str

    def __repr__(self):
        return f'TextDoc({len(self.body)}): {str(self.body)[:65]} ...'


@dataclass
class TextDocEmbedded:
    body: list

    def __repr__(self):
        return f'TextDocEmbedded(,{len(self.body)}): {str(self.body)[:65]} ...]'
