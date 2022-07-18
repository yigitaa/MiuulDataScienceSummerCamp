#integers
from builtins import dict

x = 46

#liste
x = ["btc", "eth", "xrp"]

#sözlük {"key" : value}
x = {"name": "Peter", "Age": 36}

#tuple
x = ("python", "ml", "ds")

#set (küme)
x = {"python", "ml", "ds"}

# Liste(list), sözlük(dictionary), set ve tuple aynı zamanda Python Collections (Arrays) olarak tanımlanır.

name = "yido"

name[0:2] # 2'ye kadar git. 2 hariç.


name = "yido"
dir(str)
type(len)
len(name)

"foo".startswith("F")
########################
# Liste (List)
########################

notes = [1, 2, 3, 4]
type(notes)
notes = ["1", "2", "3", "4"]

len(notes)

# append: eleman ekler
######################
notes.append(100)
notes


notes.pop(0) #eleman siler

notes.insert(2, 99) #2. indexe 99 ekle

######################
# Sözlük
######################

#key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dict2 = {"REG": ["RMSE", 10],
         "LOG": ["MSE", 20],
         "CART": ["SSE", 30]}

dict2["CART"]
dict2["CART"][1]

# Key sorgulama
"REG" in dictionary

#Key'e göre value sorgulama
dictionary["REG"]
dictionary.get("REG")

#Value değiştirmek
dictionary["REG"] = ["YSA", 10]

#tüm key'lere erişmek
dictionary.keys()
dictionary.values()

#tüm çiftleri tuple halinde listeye çevirme
dictionary.items()

#key-value değerini güncellemek
dictionary.update({"REG": 11})

###############################
# Tuple (demet)
# sıralıdır (elemanlara erişilebilir)
# kapsayıcı (birden fazla veri yapısını saklayabiliyor)
# değiştirilemez
###############################

t = ("john", "mark", 1, 2)
type(t)
t[0]
t[0:3]
# t[0] = 99 <<< tuple'ı değiştiremezsin

#############################
# Set
# değiştirilebilir
# sırasız + eşsiz
# kapsayıcı
#############################
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

#set1'de olup set2'de olmayanlar.
set1.difference(set2)
#set2'de olup set1'de olmayanlar.
set2.difference(set1)
#iki kümede de birbirinde olmayan elemanlar.
set1.symmetric_difference(set2)
#iki kümenin kesişimi
set1.intersection(set2)
set2.intersection(set1)
#iki kümenin birleşimi
set1.union(set2)