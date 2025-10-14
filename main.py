#!/venv/bin/python
'''
learning:
https://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm
'''
from flask import Flask, render_template_string
import random, heapq
#import pygame
class Maze: 
    def __init__(self,r,c):
        self.R=r; self.C=c;
        self.mid=(r//2,c//2)
        self.B=[[1 for i in range(0,c)] for j in range(0,r)]
        self.dirs=[(-1,0),(1,0),(0,-1),(0,1)] #LRDU
    def put(self,*idxs,num): 
        for idx in idxs:
            self.B[idx[0]][idx[1]]=num
    def sum2(self,t0,t1):return (t0[0]+t1[0],t0[1]+t1[1])
    def rand(self, n):
        ret=[]
        n = min(n, self.R*self.C)
        for _ in range(n):
            t=(random.randint(0,self.R-1),random.randint(0,self.C-1));
            while (t in ret): t=(random.randint(0,self.R-1),random.randint(0,self.C-1)); 
            ret.append(t);
            #for i in ret: self.B[i[0]][i[1]]=1

    def djik(self,start,end): #0:empty tile, 1: blocked; -1: target
        #self.put((self.C-1,self.r-1),1)
        self.put(self.mid,'x'); self.put(end,-1); #self.M.print()
        heap,visited=[],set() 
        heapq.heappush(heap,(0,start,[start])) #key:(cost,pos,path)
        while heap:
            cost,(R,C),path=heapq.heappop(heap)
            for D in self.dirs: 
                r,c=R+D[0],C+D[1] 
                if not (0<=r<self.R and 0<=c<self.C) or self.B[r][c] or (r,c) in visited: continue
                if (R_,C_)==end: return cost,path+[end]
                visited.add((R_,C_))
                heapq.heappush(heap,(cost+1,(R_,C_),path+[(R_,C_)]))
        return -1
    def __str__(self): return '\n'.join(str(self.B[r]) for r in range(self.R))

    def ns(self,n): return [(a+c, b+d) for ((a,b),(c,d)) in list(zip(self.dirs, [n]*4))] #neighbors
    def fs(self,n,fs): 
       f=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and self.B[rc[0]][rc[1]] and n not in fs; 
       return list(filter(f,self.ns(n)))

    def fns(self,n,fs): #forntier neighbors
        #print(n)#check if it is wall and not alr in frontiers
        ret=[]
        if n[0]+1<self.R and self.B[n[0]+1][n[1]] and ((r:=(n[0]+1, n[1])) not in fs):ret.append(r)
        if n[0]-1>=0 and self.B[n[0]-1][n[1]] and ((r:=(n[0]-1, n[1])) not in fs):ret.append(r)
        if n[1]+1<self.C and self.B[n[0]][n[1]+1] and ((r:=(n[0], n[1]+1)) not in fs):ret.append(r)
        if n[1]-1>=0 and self.B[n[0]][n[1]-1] and ((r:=(n[0], n[1]-1)) not in fs):ret.append(r)
        return ret


    def map(self,p,f): #prims algo,p=path
        self.B[p[0]][p[1]]=0 #gen from here 
        fil=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and not self.B[rc[0]][rc[1]]
        while(len(f)):
            fc=f.pop(random.randint(0,len(f)-1))
            #print(list(filter(fil,self.ns(fc))))
            #if ((fc[0] in (0,self.R-1)) or (fc[1] in (0,self.C-1))): break #to bounds, too soon
            if len(list(filter(fil,self.ns(fc))))==1:#dig if ex 1 neighboring dug tile 
                self.B[fc[0]][fc[1]]=0;
                for _ in self.fns(fc,f):f.append(_)

        side=random.randint(0,3)
        match side:
            case 0: 
                while(self.B[1][(s:=random.randint(0,self.C-1))]!=0):s+=1
                s=(0,s)
            case 1: 
                while(self.B[s:=random.randint(0,self.R-1)][-2]!=0):s+=1
                s=(s,self.C-1)
            case 2: 
                while(self.B[-2][(s:=random.randint(0,self.R-1))])!=0:s+=1
                s=(self.R-1,s)
            case 3: 
                while(self.B[s:=random.randint(0,self.R-1)][1]!=0):s+=1
                s=(s,0)
        for _ in range(self.C):
            self.B[0][_]=1
            self.B[-1][_]=1
        for _ in range(self.R):
            self.B[_][0]=1
            self.B[_][-1]=1
        self.B[s[0]][s[1]]=1
        

        return p 


dims=(20,50)
M = Maze(*dims)#print(M.fns(M.mid,()))
M.map(M.mid,M.fns(M.mid,()))
print(M)

def run():
    app = Flask(__name__)
    @app.route("/")
    def home():
        html=''
        for r in range(M.R):
            #for c in range(M.C): html+=f'''<c style="color:{["black","red"][(r,c) in P]}">{M.B[r][c]}</c>'''
            for c in range(M.C): html+=f'''<c>{M.B[r][c]}</c>'''
            html+='<br>'
        return html
    app.run(debug=True)
