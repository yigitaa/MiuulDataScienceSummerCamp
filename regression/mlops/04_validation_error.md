- Import ValidationError  
` from pydantic import BaseModel, ValidationError `  

- Create try except block  

```commandline
try:
    student1 = Student(**student1_data)

    print(student1)
except ValidationError as e:
    print(e.json())
```

- Change student1_data
` "classes": ["Information and Ent", "Physics"] `

- Run
```commandline
[
  {
    "loc": [
      "classes",
      0
    ],
    "msg": "value is not a valid enumeration member; permitted: 'Physics I', 'Calculational Methods In Physics', 'Information and Entropy'",
    "type": "type_error.enum",
    "ctx": {
      "enum_values": [
        "Physics I",
        "Calculational Methods In Physics",
        "Information and Entropy"
      ]
    }
  },
  {
    "loc": [
      "classes",
      1
    ],
    "msg": "value is not a valid enumeration member; permitted: 'Physics I', 'Calculational Methods In Physics', 'Information and Entropy'",
    "type": "type_error.enum",
    "ctx": {
      "enum_values": [
        "Physics I",
        "Calculational Methods In Physics",
        "Information and Entropy"
      ]
    }
  }
]

```

- Note the good error explanations.