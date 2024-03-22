from dataclasses import dataclass

@dataclass
class Error:
    message: str

@dataclass
class Warning:
    message: str

Verification = Error | Warning