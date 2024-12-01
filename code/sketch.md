class StaticAlgoNx: Takes in a base nxgraph G along with parameters and creates an nxgraph (or graphs) as needed.
    init and whatever is needed for init.

    Maintains a dict[str, nxGraph] that is semantically useful for visualization later on

    NO VISUALIZATION HERE.


class DynamicAlgoNx: Extends StaticAlgoNx.

    make_update_generator(is_add, u, v):
        Returns a generator which on each next() call, performs one microstep. Each next() call only performs a micro-step.

    step()
        Performs a microstep and yields a list of nodes and edges accessed and mutated. 
    
    update_complete():
        Run all remaining steps till the update is over, and return a list of all the nodes and edges mutated in that whole update step.

class StaticAlgoFigs: Takes in a StaticAlgoNx and a dict of str->figs. (Entries can be None if graph vis not needed for that graph).

   Init makes and draws the traces. That's about it. It is free to modify 'pos' as needed, and add attributes as needed, to the nxgraphs.

class DynamicAlgoFigs: Takes in a DynamicAlgoNx and a dict of str->figs (Entries can be None if graph vis not needed for that graph) 

Init as in StaticAlgoFigs.

make_update_generator, step and update_complete all correspond to the same as in DynamicAlgoNx.


Above this can be... main, I guess. 


