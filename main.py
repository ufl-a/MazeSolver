#!/venv/bin/python
from flask import Flask, render_template_string
import random
class Maze: 
    '''
    0:empty tile, 1: blocked; -1: we want to get to
    '''
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
    def place(self,idx,num): self.B[idx[0]][idx[1]]=num
    def djik():
        pass
        
app = Flask(__name__)
M=Maze(10, 10)
M.rand(1)
@app.route("/")
def home():
	M.pb()
	html = "<pre>" + "\n".join(" ".join(str(c) for c in row) for row in M.B) + "</pre>"
	return html

app.run(debug=True)


