- Import Field  
` from pydantic import BaseModel, ValidationError, Field `

- Add Field to Student class
```commandline
class Student(BaseModel):
    name: str=Field(default=None, max_length=50, min_length=3)
    email: str=Field(default=None, max_length=50, min_length=3)
    faculty: str=Field(default=None, max_length=50, min_length=3)
    classes: List[Classes]
    grade: int=Field(default=1, gt=0, lt=5)
```

- Update student1_data 
```commandline
student1_data = {
    "name": "student1",
    "email": "user1@example.com",
    "faculty": "Physics",
    "classes": [Classes.PHYS101, Classes.PHYS150],
    "grade": 6
}
```

- Run and see validation error for grade 6
```commandline
[
  {
    "loc": [
      "grade"
    ],
    "msg": "ensure this value is less than 5",
    "type": "value_error.number.not_lt",
    "ctx": {
      "limit_value": 5
    }
  }
]
```

- Access just fields and see autocomplete.
```commandline
print(student1.name)
print(student1.email)
print(student1.grade)
```

- You can also print as dictionary
`     print(student1.dict()) `

- Or schema
```commandline
print(student1.schema())

{'title': 'Student', 'type': 'object', 'properties': {'name': {'title': 'Name', 'maxLength': 50, 'minLength': 3, 'type': 'string'}, 'email': {'title': 'Email', 'maxLength': 50, 'minLength': 3, 'type': 'string'}, 'faculty': {'title': 'Faculty', 'maxLength': 50, 'minLength': 3, 'type': 'string'}, 'classes': {'type': 'array', 'items': {'$ref': '#/definitions/Classes'}}, 'grade': {'title': 'Grade', 'default': 1, 'exclusiveMinimum': 0, 'exclusiveMaximum': 5, 'type': 'integer'}}, 'required': ['classes'], 'definitions': {'Classes': {'title': 'Classes', 'description': 'An enumeration.', 'enum': ['Physics I', 'Calculational Methods In Physics', 'Information and Entropy'], 'type': 'string'}}}

```

- Json
```commandline
schema = student1.schema()
print(json.dumps(schema))
  
{"title": "Student", "type": "object", "properties": {"name": {"title": "Name", "maxLength": 50, "minLength": 3, "type": "string"}, "email": {"title": "Email", "maxLength": 50, "minLength": 3, "type": "string"}, "faculty": {"title": "Faculty", "maxLength": 50, "minLength": 3, "type": "string"}, "classes": {"type": "array", "items": {"$ref": "#/definitions/Classes"}}, "grade": {"title": "Grade", "default": 1, "exclusiveMinimum": 0, "exclusiveMaximum": 5, "type": "integer"}}, "required": ["classes"], "definitions": {"Classes": {"title": "Classes", "description": "An enumeration.", "enum": ["Physics I", "Calculational Methods In Physics", "Information and Entropy"], "type": "string"}}}

```
Don't underestimate  and think ' and " are same thing.

