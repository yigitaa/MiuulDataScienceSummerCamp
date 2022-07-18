#virtual env nedir:
#pip paket yönetimi, conda paket + sanal ortam
#pop: indexe göre siler
#insert: indexe ekler
#key sorgulamada value sorarsan false döner

def hesapla(celcius):
    kelvin = celcius + 273
    print(kelvin)

hesapla(10)

def yaslimit(yas):
    if yas < 18:
        a=print("giremenko")
        return a
    else:
        b=print("hg")
        return b

yaslimit(17)

ders = "Matematik"
puan = 70
def dersKontrol(ders, puan):
    if ders == "Matematik":
        print("mat sınavı açıklandı")
        if puan > 65:
            print("geçtin")
        else:
            print("kaldın")
    else:
        print("mat sınavı açıklanmadı")

dersKontrol(ders,puan)



A = [10,11,12,13,14,15,16]
B= []
def move(liste1, liste2):
    for i in liste1:
        liste2.append(i)

    liste1 = []
    return liste1, liste2

move(A, B)

unq = [1,1,2,3,3,4,4,4,5,6,7,7,8,9,0,0]
def nonUnique(a):
    myset = set(a)
    return myset
nonUnique(unq)


def alternating(string):
    new_string = ""

    for idx in range(len(string)):
        if idx % 2 == 0:
            new_string += string[idx].upper()
        else:
            new_string += string[idx].lower()
    return new_string

alternating("hi my name is john and i am learning python")


def function(*args):
    num = 0

    for i in args:
        num += i # num = num+i

    return num

fruits = ["apple", "pear", "banana", "kiwi", "orange"]
a = [idx for idx in fruits if len(idx) < 5]
a
###########
