#!/venv/bin/python
from flask import Flask, render_template_string
import random, heapq
class Maze: 
    def __init__(self,r,c):
        self.R=r; self.C=c;
        self.B=[[0 for i in range(0,c)] for j in range(0,r)]
    def print(self): 
        for i in range(self.R): print(self.B[i])
    def rand(self, n):
        ret=[]
        n = min(n, self.R*self.C)
        for _ in range(n):
            t=(random.randint(0,self.R-1),random.randint(0,self.C-1));
            while (t in ret): t=(random.randint(0,self.R-1),random.randint(0,self.C-1)); 
            ret.append(t);
            for i in ret: self.B[i[0]][i[1]]=1
    def place(self,idx,num): self.B[idx[0]][idx[1]]=num

    def djik(self,start,end): #0:empty tile, 1: blocked; -1: target
        #self.place((self.C-1,self.r-1),1)
        self.place(start,'x'); self.place(end,-1); #self.M.print()
        DIRS=[(-1,0),(1,0),(0,-1),(0,1)] #LRDU
        heap,visited=[],set() 
        heapq.heappush(heap,(0,start,[start])) #key:(cost,pos,path)
        while heap:
            cost,(R,C),path=heapq.heappop(heap)
            for D in DIRS: 
                R_,C_=R+D[0],C+D[1]
                if (R_>=self.R or C_>=self.C or R_<0 or C_<0) or (self.B[R_][C_]==1) or ((R_,C_) in visited): continue
                if (R_,C_)==end: return cost,path+[end]
                visited.add((R_,C_))
                heapq.heappush(heap,(cost+1,(R_,C_),path+[(R_,C_)]))


        
def run():
    M=Maze(10, 10); M.rand(1);
    _,P=M.djik((0,0),(6,6))
    app = Flask(__name__)
    @app.route("/")
    def home():
        html=''
        for r in range(M.R):
            for c in range(M.C): html+=f'''<c style="color:{["black","red"][(r,c) in P]}">{M.B[r][c]}</c>'''
            html+='<br>'
        return html
    app.run(debug=True)

run()
