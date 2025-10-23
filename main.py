#!/venv/bin/python
'''
learning:
https://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm
'''

from flask import Flask, render_template_string
import random, heapq,pygame,sys

class Maze: 
    def __init__(self,r,c):
        self.R=r; self.C=c;
        self.mid=(r//2,c//2)
        self.B=[[1 for i in range(0,c)] for j in range(0,r)]
        self.dirs=[(-1,0),(1,0),(0,-1),(0,1)] #LRDU
        self.end=0
    def __str__(self): return '\n'.join(str(self.B[r]) for r in range(self.R))
    def put(self,idx,num): self.B[idx[0]][idx[1]]=num
    def sum2(self,t0,t1):return (t0[0]+t1[0],t0[1]+t1[1])

    def gen_djik(self,src,dest):
        ys,heap,visit,cost=[],[],set(),0
        heapq.heappush(heap,(cost,src,[]))
        filt=lambda rc:(0<=rc[0]<self.R) and (0<=rc[1]<self.C) and (not self.B[rc[0]][rc[1]]) and ((rc[0],rc[1]) not in visit)
        while len(heap): 
           (cost,pos,path)=heapq.heappop(heap) 
           ys=[]
           for sq in (ns:=filter(filt,self.ns(pos))):
               if sq==dest: return (cost,path+[dest],visit)
               visit.add(sq)
               ys.append(sq)
               heapq.heappush(heap,(cost+1,sq,path+[sq]))
           #print(heap)
           ys.append(pos)
           yield ys 
        return -1,[],[]

    def gen_star(self,src=None,dest=None): 
        if src is None: src = self.mid
        if dest is None: dest = self.end
        print(src,dest)
        heur=lambda xy:( (xy[1][1]-xy[0][1])**2 + (xy[1][0]-xy[0][0])**2 )**(1/2) #euclidian dist
        op,cl,path,pars=[(0,0,src,None)],set(),[],{src:None} #op: (real,heur,node,par)
        fil=lambda rc:(0<=rc[0]<self.R) and (0<=rc[1]<self.C) and (not self.B[rc[0]][rc[1]] and rc not in cl)
        while len(op):
            top=heapq.heappop(op)#;print(top[1])
            ys=[]
            if top[2]==dest:
                ret,r=[],pars[dest]
                while r!=None: ret.append(r);r=pars[r]
                return ret
            for sq in filter(fil,self.ns(top[2])):
                h=heur((dest,sq))
                heapq.heappush(op,(top[0]+1+h,h,sq,top[2])); 
                pars[sq]=top[2]
                ys.append(sq)
            ys.append(top[2])#;print(ys)
            yield ys
            cl.add(top[2])
        return list(cl)

    def ns(self,n): return [(a+c, b+d) for ((a,b),(c,d)) in list(zip(self.dirs, [n]*4))] #neighbors
    #def fs(self,n,fs): 
    #   f=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and self.B[rc[0]][rc[1]] and n not in fs; 
    #   return list(filter(f,self.ns(n)))

    def fns(self,n,fs): #frontier neighbors, used for generating maze. (better)
        #print(n) #check if it is wall and not alr in frontiers
        ret=[]
        if n[0]+1<self.R and self.B[n[0]+1][n[1]] and ((r:=(n[0]+1, n[1])) not in fs):ret.append(r)
        if n[0]-1>=0 and self.B[n[0]-1][n[1]] and ((r:=(n[0]-1, n[1])) not in fs):ret.append(r)
        if n[1]+1<self.C and self.B[n[0]][n[1]+1] and ((r:=(n[0], n[1]+1)) not in fs):ret.append(r)
        if n[1]-1>=0 and self.B[n[0]][n[1]-1] and ((r:=(n[0], n[1]-1)) not in fs):ret.append(r)
        return ret


    def map(self,p,f): #prims algo,p=path,f=frontiers
        path={p:None}
        self.B[p[0]][p[1]]=0 #gen from here 
        fil=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and not self.B[rc[0]][rc[1]]
        while(len(f)):
            fc=f.pop(random.randint(0,len(f)-1))
            #print(list(filter(fil,self.ns(fc))))
            #if ((fc[0] in (0,self.R-1)) or (fc[1] in (0,self.C-1))): break #to bounds,too soon
            if len(dug:=(list(filter(fil,self.ns(fc)))))==1:#dig if ex 1 neighboring dug tile, 'dug' holds this neighbor
                self.B[fc[0]][fc[1]]=0;
                path[(fc[0],fc[1])]=dug[0] 
                for _ in self.fns(fc,f):f.append(_)
        match (side:=random.randint(0,3)):
            case 0: 
                while(self.B[0][(s:=random.randint(0,self.C-1))]):s+=1
                s=(0,s)
            case 1: 
                while(self.B[s:=random.randint(0,self.R-1)][-1]):s+=1
                s=(s,self.C-1)
            case 2: 
                while(self.B[-1][(s:=random.randint(0,self.R-1))]):s+=1
                s=(self.R-1,s)
            case 3: 
                while(self.B[s:=random.randint(0,self.R-1)][0]):s+=1
                s=(s,0)
        #self.B[s[0]][s[1]]=0; print("end tile:   ",s)
        self.end,s_=s,s
        ret=[s]#;print(s)
        while ((r:=path[s_])!=None): ret.append(r);s_=path[s_]
        return s,ret

class sprite:
    def ast(scn,clr,mid,px,wid):
        to,(x,y)=px//4,mid
        tos=[((x,y-to),(x,y+to)),
             ((x-to,y),(x+to,y)),
             ((x-to,y-to),(x+to,y+to)), 
             ((x-to,y+to),(x+to,y-to)),
            ]
        for (a,b) in tos: pygame.draw.line(scn,clr,a,b,wid)

def render(M,d,s,scn,px):
    for r in range(M.R):
        for c in range(M.C):
            px_=pygame.Rect(c*px,r*px,px,px)
            col=(0xff,0xff,0xff) if M.B[r][c] else (0,0,0) if (r,c) not in d.union(s) else (0xee,0xee,0)
            pygame.draw.rect(scn,col,px_)
    for (r, c) in s:sprite.ast(scn, 0xffffff,((c+0.5)*px,(r+0.5)*px),px,2)


def main():
    pygame.init()
    dims,px,rn=(20,20),40,1
    M=Maze(*dims)#;print(M.fns(M.mid,()))
    F,P=M.map(M.mid,M.fns(M.mid,())) 
    #P0,P1=M.djik(M.mid,F)[1:]
    djik=M.gen_djik(M.mid,F)#;print(next(djik))
    star=M.gen_star(M.mid,F)#;print("\t",next(star))
    scn=pygame.display.set_mode((M.C*px,M.R*px))
    clk=pygame.time.Clock()

    d,s=set(),set()
    while rn:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: rn=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_w: 
                    for _ in next(djik): d.add(_)
                    for _ in next(star): s.add(_)
        scn.fill((50, 50, 50))
        render(M,d,s,scn,px)
        pygame.display.flip()
        clk.tick(30)
    pygame.quit()
    sys.exit()

if __name__=="__main__": main()
