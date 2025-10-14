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
    #def fs(self,n,fs): f=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and self.B[rc[0]][rc[1]] and n not in fs; 
    #   return list(filter(f,self.ns(n)))

    def fns(self,n,fs): #forntier neighbors
        print(n)
        ret=[]
        if n[0]+1<self.R and self.B[n[0]+1][n[1]] and ((r:=(n[0]+1, n[1])) not in fs):ret.append(r)
        if n[0]-1>=0 and self.B[n[0]-1][n[1]] and ((r:=(n[0]-1, n[1])) not in fs):ret.append(r)
        if n[1]+1<self.C and self.B[n[0]][n[1]+1] and ((r:=(n[0], n[1]+1)) not in fs):ret.append(r)
        if n[1]-1>=0 and self.B[n[0]][n[1]-1] and ((r:=(n[0], n[1]-1)) not in fs):ret.append(r)
        return ret


    def map(self,p,f): #prims,p=point,f=frontier
        while(len(f)):
            p.append(fc:=f.pop(random.randint(0,len(f)-1)))
            self.B[fc[0]][fc[1]]=0
            if ((fc[0] in (0,self.R-1)) or (fc[1] in (0,self.C-1))): break #to bounds
            f+=self.fns(fc,f)


M=Maze(10,10);
print(M.fns(M.mid,()))
M.map([M.mid],M.fns(M.mid,()))
print(M)

def run():
    _,P=M.djik(M.mid,(9,9))
    app = Flask(__name__)
    @app.route("/")
    def home():
        html=''
        for r in range(M.R):
            for c in range(M.C): html+=f'''<c style="color:{["black","red"][(r,c) in P]}">{M.B[r][c]}</c>'''
            html+='<br>'
        return html
    app.run(debug=True)

