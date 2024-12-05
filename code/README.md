Demo for dynamic graph algorithms (Current goal: Roditty & Zwick's Dynamic APSP Algorithm)

----

In graphfig:

(What do you need to do to concretize these classes? Read below.)

StatAlgo:

    Does nothing except create vars. Subclass for more. Suitable for BFS etc.
    Maintain nx graph(s) and store attributes in them. Query not implemented. 

    init:                           CALL    MOD
    query:                          NOTIMPLEMENTED

StatVis:

    Adds traces to figs using the nx graphs of a statalgo.

    init:                           CALL    AS_IS
        Initializes variables.

    default_init_edge_visdict:      PRIV    MOD
    default_init_node_visdict:      PRIV    MOD

        visdict == the visual attributes in an nxgraph's (node or edge)'s dict
        instantiate the visdict for an edge/node -- EXCLUDING XY POSITION
        MOD to alter the way nodes/edges are default displayed (color, text, hovertext)

    default_init_nx_layout:         PRIV    MOD
        Add the POSITION to the visdicts of all currently added nodes/edges

    vis_init_visdicts:              PRIV    AS_IS
    vis_add_traces:                 PRIV    AS_IS
    vis_add_edge:                   PRIV    AS_IS 
    vis_add_node:                   PRIV    AS_IS
    
    vis_init_all                    CALL    AS_IS
        Calls other methods to prepare and add the traces to figures.

Typical flow for a StatAlgo subclass with StatVis:

StatAlgo:
1. Implement alg in init.
2. Implement query.

StatVis:
1. Outside, create a figs_dict as needed.
2. Modify default_init_node_visdict , default_init_edge_visdict and default_init_nx_layout

-----------------------------------------------------------------------------------------

DynAlgo INHERITS StatAlgo

    init                            CALL    MOD

    assign_generator                CALL    AS_IS    
    step_all_remaining              CALL    AS_IS
    step                            CALL    AS_IS
    yieldtest_update_fn             CALL    EXAMPLE -- MAKE YOUR OWN
    example_update_fn               CALL    EXAMPLE -- MAKE YOUR OWN

    get_new_update_dict             PRIV    AS_IS
    refresh_update_dict             PRIV    AS_IS
    
    nx_add_edge                     PRIV    AS_IS
    nx_remove_edge                  PRIV    AS_IS
    nx_add_node                     PRIV    AS_IS
    nx_remove_node                  PRIV    AS_IS

DynVis INHERITS StatVis

    init                            CALL    AS_IS

    vis_delete_edge                 PRIV    AS_IS
    vis_delete_node                 PRIV    AS_IS 
    vis_update_node                 PRIV    AS_IS
    vis_update_edge                 PRIV    AS_IS

    default_add_node_to_nx_layout   CALL    EXAMPLE -- MAKE YOUR OWN
    example_vis_step                CALL    EXAMPLE -- MAKE YOUR OWN 
    yieldtest_vis_step              CALL    EXAMPLE -- MAKE YOUR OWN 


Typical flow for a childclass of DynAlgo with a corresponding childclass of DynVis

DynAlgo:

1. Implement the build stage of the algo in init.     (As you did in StaticAlgo)
2. Implement query.                                    (As you did in StaticAlgo)

3. Implement an update_fn similar to yieldtest_update_fn, in the following manner:

    At the beginning of the fn, call:
        self.update_dict = self.get_new_update_dict()

    At the beginning of each micro-step, call:
        curr_updates = self.get_new_update_dict() 

    Use self.nx_... to perform the necessary updates in the nxgraphs.
    Modify the attributes in the nxdicts as needed. NOT THE VISUAL PROPERTIES.

    For every mutated node or edge,
        curr_updates[nxgraph_id]['nodes'|'edges'].append((i, 'ADD'|'DEL'|'MOD'))

    At the end of a micro-step, 
        self.refresh_update_dict(curr_updates)
        
        yield (curr_updates, False) if this is not the last microstep
            OR
        yield (curr_updates, True) if it is

A full execution of update_fn represents one or more updates ("ADD", "MOD" or "DEL" ) in the base graph, of edges or nodes.
We frequently wish to visualize this with higher temporal resolution. More steps, smaller steps.

DynVis 

1. Outside, create a figs_dict as needed.                                                       (As in StatVis)
2. Modify default_init_node_visdict , default_init_edge_visdict and default_init_nx_layout      (As in StatVis) 
3. Implement a concrete version of skeleton_vis_step

Main 

See the __main__ section of graphfig.py