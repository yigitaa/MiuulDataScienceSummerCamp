- Let's add list type  
` from typing import List`

- Add Student class  
` classes: List[str] `

- Add student1_data  
`   "classes": ["Physics I", "Calculational Methods In Physics", "Information and Entropy"] `

- Run, output will be  
` name='student1' email='user1@example.com' faculty='Physics' classes=['Physics I', 'Calculational Methods In Physics', 'Information and Entropy'] `


```commandline
from pydantic import BaseModel
from typing import List


class Student(BaseModel):
    name: str
    email: str
    faculty: str
    classes: List[str]


student1_data = {
    "name": "student1",
    "email": "user1@example.com",
    "faculty": "Physics",
     "classes": ["Physics I", "Calculational Methods In Physics","Information and Entropy"]
}

student1 = Student(**student1_data)

print(student1)
```
