import os
def findEnroll(base):
    for root, ds, fs in os.walk(base):
        if (len(fs)) >= 5:
            for f in fs[:4]:
                fullname = root+'/'+f
                yield fullname

def findTest(base):
    for root, ds, fs in os.walk(base):
        if (len(fs)) >= 5:
            fullname = root+'/'+fs[4]
            yield fullname
            

def main():
    buffer1,buffer2 = [],[]
    base = './dataset/'
    for i in findEnroll(base):
        buffer1.append(i)
    for j in findTest(base):
        buffer2.append(j)
    return buffer1, buffer2


    

if __name__ == '__main__':
    buffer1,buffer2 = main()
    with open('enrolldata1.txt','a') as file1:
        for i in buffer1:
            file1.write(i+'\n')

    with open('testdata1.txt','a') as file2:
        for j in buffer2:
            file2.write(j+'\n')




