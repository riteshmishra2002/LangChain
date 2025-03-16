from typing import TypedDict

class person(TypedDict):
    name : str
    age : int

new_person : person = {'name':'shan', 'age': 42}

print(new_person)