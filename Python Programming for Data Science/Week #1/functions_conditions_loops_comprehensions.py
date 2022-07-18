#############################
# fonksiyonlar
#############################
print("a")
print("a", "b", sep="__")


def calculate(x):
    print(x * 2)


calculate(5)


def summer(arg1, arg2):
    print(arg1 + arg2)
summer(3, 4)


###############################
# Docstring
###############################
def summer(arg1, arg2):
 """

 Args:
     arg1: int, float
     arg2: int, float

 Returns:
    int, float

 Examples:

 Notes:

 """
    print(arg1 + arg2)


 ############################
 # Fonksiyonların Statement/Body Bölümü
 ############################

 # def function_name(parameters/arguments):
 #  statements (function body)

 def say_hi(string):
     print(string)
     print("Hi")
     print("Hello")

say_hi("memort")

def multiplication(a, b):
    c = a * b
    print(c)

multiplication(3, 5)


#girilen değerleri bir liste içinde saklayacak fonksiyon

list_store = []

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(3, 5)

##############################
#Ön Tanımlı Argümanlar/Parametreler (def param/args)
##############################

def divide(a, b):
    print(a / b)

divide(1, 2)

def say_hi(ip="sa"):
    print(ip)

say_hi()


#########################
# Ne zaman fonksiyon yazma ihtiyacımız olur?
#########################

# DRY

def calculate(varm, moisture, charge):
    print((varm+moisture)/charge)

calculate(98, 12, 78)

#######################
# Return: fonksiyon çıktılarını girdi olarak kullanmak
#######################
global_output = []
def calculate(varm, moisture, charge):
    output = (varm + moisture) / charge
    global_output.append(output)
    return output


calculate(98, 12, 78)

########################
# FOnksiyon içinde fonksiyon çağırmak
########################
def standardization(a, p):
    return a * 10 / 100 * p * p
standardization(45, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 3, 5, 12)
##########################
# Local & Global değişkenler( variables)
##########################
list_store = [1, 2]

def add_element(a, b ):
    c = a * b
    list_store.append(c)
    print(list_store)

########################
# if conditions (koşul gaming)
########################
# true-false
1 == 1
1 == 2

# if

if 1 == 1:
    print("zart")

if 1 == 2:
    print("zurt")


##########################
# for loops
##########################

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(int(salary*20/100 + salary))

    #DRY prensibi: kendini tekrar etme
##########################################
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(5000, 20)

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 1000))

############################################
# Uygulama - Mülakat Sorusu
############################################

for i in range(0, 5):
    print(i)

for i in range(len("miuul")):
    print(i)


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("Yigit Ahmet Arikok")
#examples



string = "yidoli"

def deneme(string):
    new_string = ""
    for index in range(len(string)):

        if index % 2 == 0:
            new_string += string[index].upper()
        else:
            new_string += string[index].lower()
    print(new_string)

deneme("yidoli")

######################
def deneme(string):
    list = []
    for index in range(len(string)):
        if index % 2 == 0:
            list.append(string[index].upper())
        else:
            list.append(string[index].lower())
    print(*list, sep="")
deneme("miuul gaming")

#########################################
# break - continue - while
#########################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

#while

number = 1
while number < 5:
    print(number)
    number += 1

##########################
# Enumerate: Otomatik counter/indexer iile for loop ÖNEMLİ
##########################

students = ["John", "Mark", "Vanessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

#########################
# Uygulama - Mülakat Sorusu
#########################
# divide_students fonk. yaz
# çift index a listesine, tek index b listesine
# return olarak iki listeyi birleştir tek liste olsun

students = ["John", "Mark", "Vanessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

divide_students(students)

###########################
# Alternating fonksiyonunun enumerate ile yazılması
###########################

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")

###########################
# Zip : listelerdeki elemanları index sırasına göre eşleştirir. output tuple çıkarır.
###########################

students = ["John", "Mark", "Vanessa", "Mariam"]

departments = ["math", "stat", "phys", "astro"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

###########################
# lambda(!), map(!), filter, reduce
###########################

def summer(a, b):
    return a + b

new_sum = lambda a, b: a + b

new_sum(4, 5)

#map

salaries = [1000, 2000, 3000, 4000, 5000]

list(map(lambda x: x**2, salaries))

# filter

list_store= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

######################################################
# COMPREHENSIONS
######################################################

###################
# List comprehensions
###################

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

# with list comprehension

[salary * 2 for salary in salaries]
[salary * 2 for salary in salaries if salary < 3000]
[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Vanessa", "Mariam"]
students_no = ["John", "Vanessa"]

[student.lower() if student in students_no else student.upper() for student in students]
[student.upper() if student not in students_no else student.lower() for student in students]

#########################
# Dict Comprehension !önemli
#########################

dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}
dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}
{k.upper(): v ** 2 for (k, v) in dictionary.items()}

#########################
# Uygulama - Mülakat Sorusu
#########################
# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir.
# Key'ler orji, value'lar değiştirilmiş olacak.

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

#########################
# list ve dict comprehension uygulamaları
#########################

######
# bir veri setindeki değişken isimlerini değiştirmek
######

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

#### lame metod
A = []
for col in df.columns:
    A.append(col.upper())

df.columns = A
####
##### comprehension metod
df.columns = [col.upper() for col in df.columns]
#####

# isminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.

["FLAG_"+ col for col in df.columns if "INS" in col]

df.columns = ["FLAG_"+ col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

###################################
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapmak istiyoruz.
###################################
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"] # string olmayan(nümerik) değişkenleri al
## uzun yol
soz = {}
agg_list = ["mean", "min", "max", "sum"]
for col in num_cols:
    soz[col] = agg_list
##
#kısa yol: comprehension
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head() #dataframe'den sadece nümerik sütunları çek

df[num_cols].agg(new_dict) #new_dict'teki key'leri alıp num_cols'ta aynılarını görüp fonksiyonu uygular. efsanedir.
