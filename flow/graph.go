package flow

type InstanceMap map[Address]FunctionBlock
type BlockMap map[string]FunctionBlock
type EdgeMap map[Parameter][]Parameter
type ReadyMap map[string]bool

type Graph struct {
    addr        Address
    nodes       InstanceMap
    edges       EdgeMap
    inputs      ParamMap
    outputs     ParamMap
}

func NewGraph(name string, id InstanceID, nodes InstanceMap, edges EdgeMap,
              inputs ParamMap, outputs ParamMap) FunctionBlock {
    return Graph{name: name, id: id, nodes: nodes, edges: edges, inputs: inputs, outputs: outputs}
}

// Returns copies of all parameters in FunctionBlock
func (g Graph) GetParams() (inputs ParamMap, outputs ParamMap) {
    return g.inputs, g.outputs
}

// Returns a copy of FunctionBlock's InstanceId
func (g Graph) GetAddr() Address {return g.addr}

func (g Graph) Run(inputs ParamValues,
                   outputs chan ParamValues,
                   stop chan bool,
                   err chan FlowError) {
    // Check types to ensure inputs are the type defined in input parameters
    if !g.checkTypes(inputs, g.input_types) {
      return FlowError{Ok: false, Info: "Inputs are impropper types.", Addr: g}
    }

    // Declare variables
    all_waiting   := make(InstanceMap)
    all_running   := make(InstanceMap)
    all_suspended := make(InstanceMap)
    all_inputs    := make(map[Address]ParamMap)
    all_outputs   := make(map[Address]ParamMap)
    all_data_in   := make(map[Address]ParamValues)
    all_data_out  := make(map[Address]ParamValues)
    all_data_rdy  := make(map[Address]ReadyMap)
    all_stops     := make(map[Address](chan bool))
    flow_errs     := chan FlowError
    data_flow     := chan DataOut

    // Create some easy functions
    
    // Stops all children blocks
    stopAll := func() {
        // Push stop down to all subfunctions
        for name, val := range all_stops {
            val <- true
        }
    }
    
    // Pushes an error up the pipeline
    pushError := func(info string) {
        all_errs <- FlowError{Ok: false, Info: info, Addr: g.GetAddr()}
        stopAll()
    }
    
    // Adds data to all_data_in
    handleInput := func(param Parameter, val interface{}) {
        addr := param.GetAddr()                             // Get the address of the parameter
        val, exists := all_data_in[addr]                     // Check if the addr already exists
        if !exists {                                        // If the parameter exists
            all_data_in[addr] = ParamValues{addr.name: val}  // Create new map and add addr,val
        } else {                                            // Otherwise
            all_data_in[addr][addr.name] = val               // Add addr,val to the preexisting map
        }
    }
    
    // Runs a block and moves it from waiting to running, catalogues all channels
    blockRun := func(blk FunctionBlock, f_in ParamValues) {
        addr := blk.GetAddr()                           // Get address for indexing
        f_stop := chan bool                             // Get stop channel
        blk.Run(f_in, data_flow, f_stop, flow_errs)     // Run the block and retrieve channels
        delete(all_waiting[addr])                       // Delete from waiting
        all_running[addr] = blk                         // Add to running
        all_stops[addr] = f_stop                        // Map stop channels
    }

    // Iterates through all given inputs and adds them to method's all_data_ins
    loadvars := func() {
        for name, val := range inputs {      // Iterate through the names/values given in function parameters
            param, exists := g.inputs[name]  // Lookup this parameter in the graph inputs
            if exists {                      // If the parameter does exist
                if CheckType(param, val) {  // Check the types of the parameter and value
                    handleInput(param, val) // Add the value to all_data_in
                }
            } else {                         // Otherwise
                pushError("Input parameter does not exist.")
                return
            }
        }
    }

    // Iterate through all blocks that are waiting
    // to see if all of their inputs have been set.
    // If so, it runs them...
    // Deleting them from waiting, and placing them in running.
    checkWaiting := func() (ran bool) {
        ran = false
        for addr, blk := range all_waiting {
            in_params := all_inputs[addr]
            in_vals   := all_data_in[addr]
            
            // Check if all parameters are ready
            ready := true
            f_in  := make(ParamValues)                   // Establish a variable to use as the run input
            for name, param := range all_inputs {
                val, exists := all_data_in[name]
                if !exists {ready = false; break;}       // If any does not exist, we set ready to false break
                else {
                    if CheckType(param, val) {           // Check the type
                        f_in[name] = val                 // For all others, we begin setting our input parameter
                    } else {                             // If the type is wrong, throw an error
                        pushError("Input parameter is not the right type.")
                        return
                    }
                }
            }
            
            ran = true
            blockRun(blk, f_in)  // If so, then run the block
        }
        return
    }
    
    // Adds data to all_data_out
    // Deletes from all_running and adds to all_waiting
    handleOutput := func(blk FunctionBlock, vals DataOut} {
        V := vals.Values
        addr := vals.Addr
        if CheckTypes(v, all_outputs[addr]) {
            all_data_out[addr] = v     // Set the output data
            delete(all_running[addr])  // Delete from running
            all_suspended[addr] = blk  // Add to suspended
            delete(all_stops[addr])    // Delete channels
            delete(all_errs[addr])     // ..
            delete(data_flow[addr])    // ..
        } else {
            pushError("Output parameter not the right type.")
            return
        }
    }

    // Iterate through all blocks that are waiting
    // to see if all of their inputs have been set.
    // If so, it makes it runs them.
    // Deleting them from waiting, and placing them in running.
    checkRunning := func() (found bool) {
        found = false
        select {
            case data := <- data_flow:
                handleOutput(blk, data)
                found = true
                break
            case e := <- flow_err:
                if !e.Ok {
                    pushError(e.Info)
                    return
                }
        }
        return
    }
    
    shiftData := func() (success bool) {
        success = false
        for addr, vals := range all_data_out {
            for name, param := range g.edges {
                p_in_lst := g.edges[param]               // Get the input params related to this output
                out_val  := vals[name]                   // Get the cooresponding output value in vals
                for _, p := range p_in_lst {             // Iterate through them
                    addr := p.GetAddr()                  // Get the address of the input
                    
                    // See if the slot is availible for shifting
                    availible := true 
                    pv, exists := all_data_in[addr]      // Check that it's input is empty
                    if exists {                          
                        _, filled := pv[p.GetName()]          // Or at least not filled in this spot
                        if filled {availible = false}         // If not then indicate this by inverting availible to false
                    } else {                                  // If no paramvalues structure there
                        all_data_in[addr] = make(ParamValues) // Make one. This is availible.
                    }
                    
                    // Also, check the type
                    if !CheckType(p, out_val) {availible = false}
                    else {pushError("Output parameter not the right type."); return;}
                    
                    // If it is availible to shift
                    if availible {
                        all_data_in[addr][p.GetName()] = out_val // Add it to all_data_in
                        delete(all_data_out[addr][name])         // Delete it from all_data_out
                        success = true
                    }
                }
            }
        }
        return
    }
    
    restoreBlock := func() (found bool) {
        found = false
        for addr, blk := range all_suspended {  // Iterate through all the suspended blocks
            val, exists := all_data_out[addr]   // Check if any of the outputs still exist
            if (len(val) == 0) || !exists {     // If it's empty or non-existant
                delete(all_suspended[addr])     // Delete it from suspended
                all_waiting[addr] = blk         // And add it to waiting
                found = true
            }
        }
        return
    }
    
    // Main Logic
    loadvars()
    wait := checkWaiting()
    ran  := checkRunning()

}

// Checks if all keys in params are present in values
// And that all values are of their appropriate types as labeled in in params
func (g Graph) checkTypes(values ParamValues, params ParamTypes) (ok bool) {
    var val interface{}
    for name, kind := range params {
        val, exists = values[name]
        switch x := val.(type) {
            case !exists:
                return false
            case x != kind:
                return false
        }
    }
    return true
}