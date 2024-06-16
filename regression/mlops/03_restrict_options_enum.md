- To restrict options use enum 

- Import enum    
` from enum import Enum `  


- Add Classes class with Enum
```commandline
class Classes(str, Enum):
    PHYS101 = "Physics I"
    PHYS125 = "Calculational Methods In Physics"
    PHYS150 = "Information and Entropy"
```

- Change classes type in Student class
` classes: List[Classes] `

- Add studen1_data  
` "classes": [Classes.PHYS150, Classes.PHYS101] `

- Run   
` name='student1' email='user1@example.com' faculty='Physics' classes=[<Classes.PHYS150: 'Information and Entropy'>, <Classes.PHYS101: 'Physics I'>] `  

```commandline
from pydantic import BaseModel
from typing import List
from enum import Enum


class Classes(str, Enum):
    PHYS101 = "Physics I"
    PHYS125 = "Calculational Methods In Physics"
    PHYS150 = "Information and Entropy"


class Student(BaseModel):
    name: str
    email: str
    faculty: str
    classes: List[Classes]


student1_data = {
    "name": "student1",
    "email": "user1@example.com",
    "faculty": "Physics",
    "classes": [Classes.PHYS150, Classes.PHYS101]
}

student1 = Student(**student1_data)

print(student1)
```