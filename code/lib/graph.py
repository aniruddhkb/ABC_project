class Graph:
    '''
    Undirected unweighted graphs. Store vertex list of ints as set. For each vert, store the adjacent vertices in a set
    '''
    def __init__(self,verts:list[int]|set[int],edges:list[tuple[int,int]]):
        self.verts = set(verts)
        self.vertattrs = {}
        [self.vertattrs[v] = {} for v in self.verts] 
        self.nbrs: dict[int,set[int]] = {}
        for v in self.verts:
            self.nbrs[v] = set() #neighbours
        self.edges = set()
        for (u, v) in edges:
            self.addBasicEdge(u,v,True)
        

    def __repr__(self) -> str:
        return (self.nbrs.__repr__(), self.edges.__repr__())
        
    
    def addBasicEdge(self,u:int, v:int, check_exists:bool = True)->None|bool:
        
        if(u==v):
            raise ValueError("Both vertices of given edge are identical. Self-edges not allowed.")
        uv_tuple = (min(u,v), max(u,v))
        adj_v = self.nbrs[v]
        adj_u = self.nbrs[u]
        
        if check_exists:
            u_in_adj_v = u in adj_v
            v_in_adj_u = v in adj_u
            if(u_in_adj_v and v_in_adj_u):
                if(uv_tuple not in self.edges):
                    raise ValueError("Something's wrong. self.edges inconsistent with self.nbrs")
                return True 
            elif(not u_in_adj_v and not v_in_adj_u):
                if(uv_tuple not in self.edges):
                    raise ValueError("Something's wrong. self.edges inconsistent with self.nbrs")
                self.edges.add(uv_tuple)
                adj_v.add(u)
                adj_u.add(v)
                return False 
            else:
                raise ValueError(f"u_in_v is {u_in_adj_v} and v_in_u is {v_in_adj_u}. Shouldn't be possible.")
        else:
            adj_v.add(u)
            adj_u.add(v)

    def removeBasicEdge(self,u:int, v:int, check_exists:bool = True)->None|bool:

        if(u==v):
            raise ValueError("Both vertices of given edge are identical. Self-edges not allowed.")
        adj_v = self.nbrs[v]
        adj_u = self.nbrs[u]
        if check_exists:
            u_in_adj_v = u in adj_v
            v_in_adj_u = v in adj_u
            if(u_in_adj_v and v_in_adj_u):
                adj_v.remove(u)
                adj_u.remove(v)
                return True 
            elif(not u_in_adj_v and not v_in_adj_u):
                return False 
            else:
                raise ValueError(f"u_in_v is {u_in_adj_v} and v_in_u is {v_in_adj_u}. Shouldn't be possible.")
        else:
            adj_v.remove(u)
            adj_u.remove(v)

