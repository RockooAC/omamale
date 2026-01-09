import os

class Env:
    def __init__(self, name, default=""):
        self.name = name
        self.default = default
        self.required = True if default else False

    def get(self):
        value = os.getenv(self.name)
        if not value:
            if self.required:
                value = self.default
            else:
                raise ValueError(f"Required environment variable {self.name} is not set")
        return value

