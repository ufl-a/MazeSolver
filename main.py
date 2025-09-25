#!/venv/bin/python

class Maze:
    def __init__(self,r,c):
        self.R=r; self.C=c; 
        self.B=[[0 for i in range(0,r)] for j in range(c)]
        for i in range(self.R): print(self.B[i])
        
Maze(10,10)
                


