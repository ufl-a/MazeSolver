#!/venv/bin/python
'''
Algos Used:
https://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm
'''

from flask import Flask, render_template_string
import random,heapq,pygame,sys,math

class Maze: 
    def __init__(self,r,c):
        self.R=r; self.C=c;
        self.mid=(r//2,c//2)
        self.B=[[1 for i in range(0,c)] for j in range(0,r)]
        self.dirs=[(-1,0),(1,0),(0,-1),(0,1)] #LRDU
        self.end=None
        self.path=None
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
        self.path=ret
        return s,ret

class sprite:
    def star(scn,clr,mid,px,wid):
        to,(x,y)=px//4,mid
        tos=[((x,y-to),(x,y+to)),
             ((x-to,y),(x+to,y)),
             ((x-to,y-to),(x+to,y+to)), 
             ((x-to,y+to),(x+to,y-to)),
            ]
        for (a,b) in tos: pygame.draw.line(scn,clr,a,b,wid)

def render(M,tl,d,s,scn,px,rev,zm,v=100,winW=1000,ft="arial",fts=20): 
    #render v tiles from top-left(tl)
    #ofx,ofy=(max(0,ofs[0]//px),max(0,ofs[1]//px)) #shift,start
    px=winW/v
    f=pygame.font.SysFont("consolas",20)
    strs,ofy=["Recenter",("Show" if not rev else "Hide")+" Path","Restart"],5
    for _ in strs:
        scn.blit(f.render(_,1,(0xff,0xff,0xff)),(v*px+5*10,10*ofy))
        ofy+=10
    (R,C)=(int(tl[0]),int(tl[1]))#;print(tl)
    U=d.union(s)
    #st=(max(0,int(math.floor(tl[0]))),max(0,int(math.floor(tl[1]))))
    #end=(min(M.R,st[0]+v+1),min(M.C,st[1]+v+1))
    #for r in range(st[0],end[0]): #alternatively, we could do a loop like this
    for r in range(v):
        r_=R+r #real row/col 
        for c in range(v):
            c_=C+c
            px_=pygame.Rect(c*px,r*px,px,px)
            #print(M.path);exit()
            col=(0xff,0xff,0xff) if M.B[r_][c_] \
                    else (0,0,0) if (((rc:=(r_,c_)) not in U) and (not rev or rc not in M.path)) \
                    else (0,0,0xff) if (rev and rc in M.path and rc not in d) else (0xee,0xee,0)
            if (r_,c_)==M.end: col=(0xff,0,0)
            pygame.draw.rect(scn,col,px_)
            if (r_,c_) in s:
                sprite.star(scn, 0x00ffff,((c+0.5)*px,(r+0.5)*px),px,10) #overlay A* squares 

def main():
    pygame.init();pygame.font.init()
    dims=(round(1e5**.5),round(1e5**.5)+1) #316,317
    tl,view,px=(0,0),(100,100),10
    px0=px*view[0]
    vw,vh=view[0]*px,view[1]*px
    pw=20*px #panel

    run,rev=1,0
    d,s=set(),set()
    drag,mpos=0,(0,0)
    (z0,z1)=1,1
    zmin,zmax=.5,1.5
    #dims,px,run=(20,20),40,1

    M=Maze(*dims)#;print(M.fns(M.mid,()))
    M.map(M.mid,M.fns(M.mid,())) #P0,P1=M.djik(M.mid,F)[1:]
    print('start,end:\t',M.end)
    djik,star=(M.gen_djik(M.mid,M.end),M.gen_star(M.mid,M.end))
    if (dims[0]>view[0]) and (dims[1]>view[1]): #maze-cam is 1e4
        sv=tl=((M.mid[0]-view[0]/2,M.mid[1]-view[1]/2)) #float
        #sv=tl=((M.mid[0]-view[0]//2,M.mid[1]-view[1]//2)) #this is less precise
    scn=pygame.display.set_mode((view[0]*px+pw,view[1]*px))
    pygame.display.set_caption("")
    clk=pygame.time.Clock()

    while run:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: run=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_w: 
                    try:
                        d=d.union(set(next(djik)))
                    except StopIteration:
                        pass
                    try:
                        s.union(set(next(star)))#;print(d,s)
                    except StopIteration:
                        pass
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button:
                if (epos:=event.pos)[0]<view[0]*px:
                    drag,mpos=(1,event.pos)
                elif epos[1]<10*px: tl=sv
                elif 10*px<epos[1]<20*px:
                    rev=not rev
                elif 20*px<epos[1]<30*px:
                    djik,star=(M.gen_djik(M.mid,M.end),M.gen_star(M.mid,M.end))
                    d,s=set(),set()

            elif event.type==pygame.MOUSEBUTTONUP and event.button:drag=0 #drag
            elif event.type==pygame.MOUSEMOTION and drag and ((epos:=event.pos[0])<view[1]*px):
                epos=event.pos
                delt=epos[0]-mpos[0],epos[1]-mpos[1] #view delta
                tl=(tl[0]-(delt[1]/px),tl[1]-(delt[0]/px))
                tl=(max(0,tl[0]),max(0,tl[1]))
                tl=(min(dims[0]-view[0],tl[0]),min(dims[1]-view[1],tl[1]))
                mpos=epos

            elif event.type == pygame.MOUSEWHEEL: #view min:50,max:200, (view doesn't really need to be tuple)
                if ((epos:=pygame.mouse.get_pos())[0]<px0):
                   v0=view 
                   tw0,th0=(px0/v0[0]),(px0/v0[1])
                   cs0=(tl[0]+epos[0]/tw0,tl[1]+epos[1]/th0) #cursor's square=topleft+offset at given scaling
                   v1=round(min(max(50,v0[0]*(.9 if (event.y>0) else 1.1)),200))
                   view=(v1,)*2;#print(view)
                   tw1,th1=px0/view[0],px0/view[1]
                   tl=(cs0[0]-epos[0]/tw1,cs0[1]-epos[1]/th1)
        scn.fill((0,0,0))
        d,s=d.union(set(next(djik))),s.union(set(next(star)))#;print(d,s)

        #maze,topleft,dijk,astar,scrn,pxwidth,revealedstate,zoomratio
        render(M,tl,d,s,scn,px,rev,z1,v=view[0])  

        pygame.display.flip()
        clk.tick(10)
    pygame.quit()
    sys.exit()

if __name__=="__main__": main()
