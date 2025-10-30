#!/venv/bin/python
'''
Research:
https://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm
'''

import random,pygame,sys,math,threading,time
e=sys.exit #for debug

class heapq: #mostly copied from std module; this is a minheap.
    def ripple_up(heap,pos):
        end=len(heap)
        start=pos
        inserted_item=heap[pos]
        cidx=2*pos+1
        while cidx<end:
            rcidx=cidx+1
            if rcidx<end and not (heap[cidx]<heap[rcidx]):
                cidx=rcidx
            heap[pos]=heap[cidx] #check children,and swap with the smaller 
            pos=cidx
            cidx=2*pos+1 #moving down.
        heap[pos]=inserted_item
        heapq.ripple_down(heap,start,pos) #check for further swaps upward.
    def ripple_down(heap,start,pos):#start from end.
        inserted_item=heap[pos]
        while start<pos:#adjust 'pos', where inserted_item is eventually placed
            pidx=(pos-1)>>1 #since cidx=2*pidx+1
            if (par:=heap[pidx])>inserted_item:
                heap[pos]=par 
                pos=pidx #moving up.
                continue
            break
        heap[pos]=inserted_item
    def heappush(heap,item):
        heap.append(item)
        heapq.ripple_down(heap,0,len(heap)-1)
    def heappop(heap):
        last=heap.pop()
        if len(heap):
            ret=heap[0]
            heap[0]=last
            heapq.ripple_up(heap,0)
            return ret
        return last

class Maze: 
    def __init__(self,r,c):
        self.R=r; self.C=c;
        self.mid=(r//2,c//2)
        self.B=[[1 for i in range(0,c)] for j in range(0,r)]
        self.dirs=[(-1,0),(1,0),(0,-1),(0,1)] #LRDU
        self.end=None
        self.path={}
        #self.pmask=None
    def __str__(self): return '\n'.join(str(self.B[r]) for r in range(self.R))
    def put(self,idx,num): self.B[idx[0]][idx[1]]=num
    def sum2(self,t0,t1):return (t0[0]+t1[0],t0[1]+t1[1])
    def ns(self,n): return [(a+c, b+d) for ((a,b),(c,d)) in list(zip(self.dirs, [n]*4))] #neighbors
    def ns2(self,n): return [(a+c, b+d) for ((a,b),(c,d)) in list(zip([(d0*2,d1*2) for (d0,d1) in self.dirs], [n]*4))] 
    #def fs(self,n,fs):f=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and self.B[rc[0]][rc[1]] and n not in fs;return list(filter(f,self.ns(n)))
    def fns(self,n,fs): #frontier neighbors, used for generating maze. (better)
        ret=[] #check if it is wall and not alr in frontiers
        if n[0]+1<self.R and self.B[n[0]+1][n[1]] and ((r:=(n[0]+1, n[1])) not in fs):ret.append(r)
        if n[0]-1>=0 and self.B[n[0]-1][n[1]] and ((r:=(n[0]-1, n[1])) not in fs):ret.append(r)
        if n[1]+1<self.C and self.B[n[0]][n[1]+1] and ((r:=(n[0], n[1]+1)) not in fs):ret.append(r)
        if n[1]-1>=0 and self.B[n[0]][n[1]-1] and ((r:=(n[0], n[1]-1)) not in fs):ret.append(r)
        return ret

    #~~~~~~~~~ SOLVING ~~~~~~~~~~~~~
    def gen_djik(self,src=None,dest=None):
        global fd,vd
        if src is None: src = self.mid
        if dest is None: dest = self.end
        ys,heap,visit,cost=[],[],set(),0
        heapq.heappush(heap,(cost,src,[]))
        filt=lambda rc:(0<=rc[0]<self.R) and (0<=rc[1]<self.C) and (not self.B[rc[0]][rc[1]]) and ((rc[0],rc[1]) not in visit)
        while len(heap): 
           (cost,pos,path)=heapq.heappop(heap) 
           ys=[]
           for sq in (ns:=filter(filt,self.ns(pos))):
               visit.add(sq)
               ys.append(sq)
               if sq==dest: 
                   fd=path+[dest] #return (cost,path+[dest],visit)
                   vd=len(visit)
                   yield tuple(ys)
                   return fd
               heapq.heappush(heap,(cost+1,sq,path+[sq]))
           #print(heap)
           vd=len(visit)
           yield tuple(ys)
           #ys.append(pos)
        return -1,[],[]

    def gen_star(self,src=None,dest=None): 
        global fs,heur,heurs,vs
        if src is None: src = self.mid
        if dest is None: dest = self.end;print(src,dest)
        H=heurs[heur]
        op=[((sh:=H((dest,src))),sh,src,None)]
        cl,path,pars=set(),[],{src:None} #open: ([distance from start+heur],heur,node,par) | closed (visited) | parents
        fil=lambda rc:(0<=rc[0]<self.R) and (0<=rc[1]<self.C) and (not self.B[rc[0]][rc[1]] and rc not in cl)
        while len(op):
            cost,h,node,_=heapq.heappop(op)#;print(top[1])
            ys=[] #yields, added to s()
            cl.add(node)
            if node==dest:
                ret,r=[],pars[dest]
                while r!=None: ret.append(r);r=pars[r]
                vs=len(cl)
                yield []; return ret
            delt=cost-h
            for sq in filter(fil,self.ns(node)):
                h0=H((dest,sq))
                heapq.heappush(op,(delt+h0+1,h0,sq,node)); 
                pars[sq]=node
                ys.append(sq)
            #ys.append(node)#;print(ys)
            vs=len(cl)
            yield tuple(ys)
            cl.add(node)
        return list(cl)

    #~~~~~~~~~ MAZE GEN ~~~~~~~~~~~~~
    def get_end(self,path=None,src=None): #after maze gen, select an exit, and build a path back to src.
        if src is None: src = self.mid
        if path is None: path = self.path;path[src]=None
        s=[(0,i) for i in range(self.C) if self.B[0][i]==0]
        s.extend([(i,self.C-1) for i in range(self.R) if self.B[i][-1]==0])
        s.extend([(self.R-1,i) for i in range(self.C) if self.B[-1][i]==0])
        s.extend([(i,0) for i in range(self.R) if self.B[i][0]==0])
        s=random.choice(s)
        self.end,s_=s,s
        ret=[s];#print(s);e()
        while ((r:=path[s_])!=None): ret.append(r);s_=path[s_]
        self.path=set(ret)
        #self.pmask=set([[1 if (r,c) in self.path else 0 for c in range(self.C)] for r in range(self.R)])
        return s,ret

    def map_prim(self,p,f): #prims algo,p=path,f=frontiers
        path={self.mid:None}
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
        s,r=self.get_end(path)

    def map_dfs(self,src=None):
        if src is None:src=self.mid
        self.B[src[0]][src[1]]=0
        path,vis,stk={src:None},{src},[src]
        fil=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and self.B[rc[0]][rc[1]] and rc not in vis
        fil0=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and not self.B[rc[0]][rc[1]]
        while len(stk):
            top=stk[-1];#print(top)
            if (sqs:=list(filter(fil,self.ns2(top)))):
                sq=sqs[random.randint(0,len(sqs)-1)]
                vis.add(sq)
                mid=((sq[0]+top[0])//2,(sq[1]+top[1])//2)
                self.B[sq[0]][sq[1]]=self.B[mid[0]][mid[1]]=0
                #if len(list(filter(fil0,self.ns(sq))))<=5:
                if len(list(filter(fil0,self.ns(sq))))==1:
                    path[sq]=mid;path[mid]=top
                stk.append(sq);
            else: stk.pop(-1)
        s,r=self.get_end(path)

    def map_dfsr(self,src=None): # RecursionError due to native limits (use only in arcade mode)
        if src is None:src=self.mid
        self._map_dfsr(src)
        self.get_end()
    def _map_dfsr(self,src=None):
        self.B[src[0]][src[1]]=0
        dirs=[(2*a,2*b) for (a,b) in self.dirs]
        fil0=lambda rc:(0<=rc[0]<self.R and 0<=rc[1]<self.C) and not self.B[rc[0]][rc[1]]
        random.shuffle(dirs)
        for d in dirs:
            if 0<=(s0:=(src[0]+d[0]))<self.R and 0<=(s1:=(src[1]+d[1]))<self.C and self.B[s0][s1]:
                n0,n1=d[0]//2+src[0],d[1]//2+src[1]
                if len(list(filter(fil0,self.ns((n0,n1)))))==1:
                    self.path[(n0,n1)]=src
                    self.path[(s0,s1)]=(n0,n1)
                self.B[n0][n1]=self.B[s0][s1]=0
                self._map_dfsr((s0,s1))
            
    def mapbfs(self,p,f):
        pass

class sprite:
    def star(scn,clr,mid,px,wid):
        to,(x,y)=px//4,mid
        tos=[((x,y-to),(x,y+to)),
             ((x-to,y),(x+to,y)),
             ((x-to,y-to),(x+to,y+to)), 
             ((x-to,y+to),(x+to,y-to)),
            ]
        for (a,b) in tos: pygame.draw.line(scn,clr,a,b,wid)
    def arrow_pair(scn,clr,mid,pad,px,wid):
        px,(x,y)=px,mid
        tos=[((x+pad,y),(x+pad+px,y)),
             ((x-pad,y),(x-pad-px,y)),
             ((x-pad-px,y),(x-pad-px//2,y+(px))),
             ((x-pad-px,y),(x-pad-px//2,y-(px))),
             ((x+pad+px,y),(x+pad+px//2,y+(px))),
             ((x+pad+px,y),(x+pad+px//2,y-(px))),
             ]
        for (a,b) in tos: pygame.draw.line(scn,clr,a,b,wid)

def render(M,tl,d,s,scn,px,rev,zm,v=100,ft="arial",fts=20):
    #render v tiles from top-left(tl)
    #ofx,ofy=(max(0,ofs[0]//px),max(0,ofs[1]//px)) #shift,start
    global px0,pw,rf,stop,heur,mgen,ofx,vd,vs
    f=pygame.font.SysFont("consolas",20)
    strs=["Recenter", \
          ("Show" if not rev else "Hide")+" Path","Restart",\
            "Complete" if not rch else "New Board",\
            "Take 50 Steps", \
            "Resume" if stop else "Stop",
    ]
          
    arrs=[f"Heur:{["Euclidian","Manhattan"][heur]}",f"MazeGen:{["Prims","IT-DFS","BFS"][mgen]}",]
    txt=[["DijkstraSquares:",str(vd)],["A*Squares:",str(vs)]]

    px=10#consant px for panel
    ofy=5*px
    for _ in strs:
        scn.blit(f.render(_,1,(0xff,0xff,0xff)),(ofx,ofy));ofy+=5*px
    for _ in arrs:
        sprite.arrow_pair(scn,0xffffff,(px0+pw//2,ofy),20,10,4);ofy+=20
        scn.blit(f.render(_,1,(0xff,0xff,0xff)),(ofx,ofy));ofy+=50
    if rch:
        for (i,_) in enumerate(txt):
            c= [(0xff,0,0),(0,0xff,0)][i]
            for a,b in [_]:
                scn.blit(f.render(a,1,c),(ofx, ofy));ofy+=5*px
                scn.blit(f.render(b,1,c),(ofx, ofy));ofy+=5*px
    (R,C)=(int(tl[0]),int(tl[1]))#;print(tl)
    U=d.union(s)
    #st=(max(0,int(math.floor(tl[0]))),max(0,int(math.floor(tl[1]))))
    #end=(min(M.R,st[0]+v+1),min(M.C,st[1]+v+1))
    #for r in range(st[0],end[0]): #alternatively, we could do a loop like this

        # ~~~~ VARIABLE PX [MAZE] ~~~~~~~~~
    px=px0/v #variable px (for maze)
    for r in range(v):
        r_=R+r #real row/col 
        for c in range(v):
            c_=C+c
            px_=pygame.Rect(c*px,r*px,px,px)
            col=(0xff,0xff,0xff) if M.B[r_][c_] \
                    else (0,0,0) if (((rc:=(r_,c_)) not in U) and (not rev or rc not in M.path)) \
                    else (0xff,0x10,0xf0) if (rev and rc in M.path and rc not in d) else (0xee,0xee,0)

            if (r_,c_)==M.mid: col=(0x80,0,0x80)
            if (r_,c_)==M.end: col=(0xff,0,0)
            pygame.draw.rect(scn,col,px_)
            if (r_,c_) in s:
                sprite.star(scn, 0x00ffff,((c+0.5)*px,(r+0.5)*px),px,10) #overlay A* squares 




def main():
    #~~~~~~~ VARS ~~~~~~~~~~~~~~
    global d,s,rch,fd,px0,stop,pw,heur,heurs,mgen,ofx,ofy,vd,vs
    #d-tiles,s-tiles,reached end,found path-dijk,display-window width,
    #stop execution,panel width,heuristic index(int),heur lambda list,
    #display offset(x,y),

    dbug=len(sys.argv)>1
    solver_thread = None
    pygame.init();pygame.font.init()

    dims=(round(1e5**.5),round(1e5**.5)+1) if not dbug else (20,20) #316,317
    tl,view,px=(0,0),(100,100),10 # viewport:(100,100)
    px0=px*view[0]
    vw,vh=view[0]*px,view[1]*px
    vd,vs=0,0
    pw=20*px #panel
    ofx=px0+15

    heurs=[
          lambda xy:( (xy[1][1]-xy[0][1])**2 + (xy[1][0]-xy[0][0])**2 )**(1/2),
          lambda xy: (abs(xy[0][0]-xy[1][0])+abs(xy[0][1]-xy[1][1])),
        ]
    mgens=["Prims","IT-DFS","REC-DFS","BFS"]
    heur,mgen=0,0
    run,rev,rch,stop=1,0,0,0
    d,s=set(),set()
    fd,fs=[],[] #found path
    drag,mpos=0,(0,0)
    (z0,z1)=1,1
    zmin,zmax=.5,1.5


    #~~~~~~~~~~ MAZE ~~~~~~~~~~~~~
    M=Maze(*dims)#;print(M.fns(M.mid,()))
    M.map_prim(M.mid,M.fns(M.mid,()));#print('start,end:\t',M.end)          #PRIMS
    #M.map_dfs(M.mid);#print('start,end:\t',M.end)                            #DFS
    #M.map_dfsr(M.mid);#print('start,end:\t',M.end)                          #DFSR
    djik,star=(M.gen_djik(M.mid,M.end),M.gen_star(M.mid,M.end))
    if (dims[0]>view[0]) and (dims[1]>view[1]): #maze-cam is 1e4
        sv=tl=((M.mid[0]-view[0]/2,M.mid[1]-view[1]/2)) #float

    scn=pygame.display.set_mode((view[0]*px+pw,view[1]*px))
    pygame.display.set_caption("")

    res_E=threading.Event(); #cmpl_T thread for COMPLETE task
    res_E.set()
    cmpl_T=None
    def cmpl(): 
        global rch
        while (not rch):
            for _ in range(10):
                res_E.wait()
                if rch or res_E.is_set():
                    break
            step()
            time.sleep(0.001)

    def step(): #if reached, show path
        global d,s,rch,vd,vs
        try:
            d=d.union(set((out_d:=(next(djik)))));
        except StopIteration:
            rch=1;return
        try:
            s=s.union(set((out_s:=(next(star)))));
        except StopIteration:
            rch=1;return

    clk=pygame.time.Clock()
    while run:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: 
                run=False
            elif event.type==pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if event.key==pygame.K_w:step()
                if keys[pygame.K_LCTRL] and event.key == pygame.K_c:run=0
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button:
                if (epos:=event.pos)[0]<px0:
                    drag,mpos=(1,event.pos)

                # ~~~~~~~~~~  panel functions
                else:       
                    #if epos[1]<5*px: tl=sv 
                    if 5*px<epos[1]<10*px: # recenter
                        tl=sv
                    if 10*px<epos[1]<15*px: # showpath
                        rev=not rev
                    if 15*px<epos[1]<20*px: #restart/compl.
                        djik,star=(M.gen_djik(M.mid,M.end),M.gen_star(M.mid,M.end))
                        d,s,rch,stop=set(),set(),0,0
                        if cmpl_T and cmpl_T.is_alive():
                            cmpl_T.join(timeout=.1)
                            cmpl_T=None

                        cmpl_T=None
                    if 20*px<epos[1]<25*px: #regen/complete (threadcall)
                        if rch:
                            res_E.clear();stop=1
                            if cmpl_T is None and cmpl_T.is_alive():
                                cmpl_T.join(timeout=.1)
                                cmpl_T=None
                            M.B=[[1 for j in range(0,M.C)] for i in range(0,M.R)]
                            match mgen:
                                case 0: M.map_prim(M.mid,M.fns(M.mid,()))
                                case 1: M.map_dfs(M.mid)
                                case 3: M.map_bfs(M.mid)
                            djik,star=(M.gen_djik(M.mid,M.end),M.gen_star(M.mid,M.end))
                            d,s,rch,stop=set(),set(),0,0
                            continue
                        if cmpl_T is None or not cmpl_T.is_alive() and not stop:
                            res_E.set()
                            cmpl_T=threading.Thread(target=cmpl,daemon=True)
                            cmpl_T.start()

                    if 25*px<epos[1]<30*px: # N-steps
                        for _ in range(50): step()
                    if 30*px<epos[1]<32*px and (cmpl_T is not None): # unset resume. Complete thread awaits (re)set.
                        stop=not stop
                        if stop: res_E.clear() 
                        else: res_E.set() 
                    if 35*px<epos[1]<40*px: 
                        heur=(heur+(-1 if epos[0]<ofx else 1))%len(heurs)
                    if 40*px<epos[1]<45*px: 
                        mgen=(mgen+(-1 if epos[0]<ofx else 1))%len(mgens)

            #~~~~~~drag and scroll~~~~~~~
            elif event.type==pygame.MOUSEBUTTONUP and event.button:drag=0
            elif event.type==pygame.MOUSEMOTION and drag and ((epos:=event.pos[0])<px0):
                epos=event.pos
                delt=epos[0]-mpos[0],epos[1]-mpos[1] #view delta
                tl=(tl[0]-(delt[1]/px),tl[1]-(delt[0]/px))
                tl=(max(0,tl[0]),max(0,tl[1]))
                tl=(min(dims[0]-(mxv:=max(view)),tl[0]),min(dims[1]-mxv,tl[1]))
                mpos=epos

            elif event.type == pygame.MOUSEWHEEL: #view min:50,max:200
                #method of zooming while keeping cursor on a square
                if ((epos:=pygame.mouse.get_pos())[0]<px0):
                    v0=view 
                    tw0,th0=(px0/v0[0]),(px0/v0[1])
                    cs0=(tl[0]+epos[0]/tw0,tl[1]+epos[1]/th0) #cursor's square=topleft+offset at given scaling
                    cs0=((min(max(0,cs0[0]),M.R)),(min(max(0,cs0[1]),M.C)))
                    #cs0=(min(max(0,cs0[0]),M.R-px0/px),min(max(0,cs0[1]),M.C-px0/px))
                    v1=round(min(max(50,v0[0]*(.9 if (event.y>0) else 1.1)),200))
                    if v0!=v1:#tilesize change
                        view=(v1,)*2;#print(view)
                        tw1,th1=px0/view[0],px0/view[1] # new square size (ss)
                        tl=(cs0[0]-epos[0]/tw1,cs0[1]-epos[1]/th1) # adjust top-left relative to cs and ss
                        tl=(max(0,min(M.R-view[0],tl[0])),(max(0,min(M.R-view[1],tl[1]))))

        scn.fill((0,0,0))
        #args: maze,topleft,dijk,astar,scrn,pxwidth,revealedstate,zoomratio
        #print(vd,vs)
        if not dbug:render(M,tl,d,s,scn,px,rev,z1,v=view[0])  
        else: render(M,tl,d,s,scn,px,rev,z1,v=20) ###DEBUG DIMS=20
        pygame.display.flip()
        clk.tick(10)
    pygame.quit()
    sys.exit()

if __name__=="__main__": main()
