#!/venv/bin/python
import random
class Maze:
    def __init__(self,r,c):
        self.R=r; self.C=c; 
        self.B=[[0 for i in range(0,c)] for j in range(0,r)]
    def pb(self): 
        for i in range(self.R): print(self.B[i])
    def rand(self, n):
        ret=[]
        n = min(n, self.R * self.C)
        for _ in range(n):
            t=(random.randint(0,self.R-1),random.randint(0,self.C-1));
            while (t in ret): t=(random.randint(0,self.R),random.randint(0,self.C)); 
            ret.append(t);
        for i in ret: self.B[i[0]][i[1]]=1


m=Maze(10,10)
m.rand(1)
m.pb()


