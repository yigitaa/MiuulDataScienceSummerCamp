```commandline
from pydantic import BaseModel


class Student(BaseModel):
    name: str
    email: str
    faculty: str


student1_data = {
    "name": "student1",
    "email": "user1@example.com",
    "faculty": "Physics"
}

student1 = Student(**student1_data)

print(student1)
# name='student1' email='user1@example.com' faculty='Physics'
```
