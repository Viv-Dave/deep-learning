def even(x):
    print("even no.: ")
    for i in range(0, x+1):
        if (i%2 ==0):
            print(i, end ="")

def odd(x):
    print("odd no.:")
    for i in range(0, x+1):
        if i%2 !=0:
            print(i, end=" ")

num = int(input("Enter a number: "))
even(num)
odd(num)