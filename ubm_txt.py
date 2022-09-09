import os
def findUbmEnroll(base):
    for root, ds, fs in os.walk(base):
        for f in fs[:4]:
            fullname = root+'/'+f
            yield fullname



def main():
    buffer = []
    base = './dataset/'
    for i in findUbmEnroll(base):
        buffer.append(i)

    return buffer


    

if __name__ == '__main__':
    buffer = main()
    with open('UBMenrolldata.txt','a') as file1:
        for i in buffer:
            file1.write(i+'\n')
