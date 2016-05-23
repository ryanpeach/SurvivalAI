package flow

type ParamMap map[string]Parameter
type MultiParamMap map[string][]Parameter
type MultiParamTypes map[string][]TypeStr
type NodeMap map[string]FunctionBlock
type EdgeMap map[*FunctionBlock][]*FunctionBlock

type Graph struct {
    name        string
    id          InstanceID
    nodes       NodeMap
    edges       EdgeMap
    inputs      MultiParamMap
    input_types ParamTypes
    outputs     ParamMap
}

// Returns a copy of FunctionBlock's name
func (g Graph) GetName() string {return g.name}

// Returns copies of all parameters in FunctionBlock
func (g Graph) GetParams() (inputs []Parameter, outputs []Parameter) {
    inputs = make([]Parameter, 0, len(g.inputs))
    for name, p_list := range g.inputs {
        append(inputs, Parameter{blockID: g.id, paramName: name, paramType: p_list[0].paramType})
    }

    outputs = make([]Parameter, 0, len(g.outputs))
    for name, p := range g.outputs {
        append(outputs, Parameter{blockID: g.id, paramName: name, paramType: p.paramType})
    }
    return
}

func (g Graph) Run(inputs ParamValues,
                  outputs chan ParamValues,
                  stop chan bool,
                  err chan FlowError) {
    // Check types to ensure inputs are the type defined in input parameters
    if !multiCheckTypes(inputs, g.inputs) {
      return FlowError{Ok: false, Info: "Inputs are impropper types.", Addr: &g}
    }

    // Declare variables
    all_waiting   := make(map[string]*FuncBlock)
    all_running   := make(map[string]*FuncBlock)
    all_inputs    := make(map[string]ParamValues)
    all_inparams  := make(map[string][]Parameter)
    all_outparams := make(map[string][]Parameter)
    all_stops     := make(map[string](chan bool))
    data_out      := make(chan DataValues)
    errs          := make(chan FlowError)

    // Iterates through all given inputs and adds them to method's all_inputs map
    // Also sets all_inparams, all_outparams, and all_blocks
    loadvars := func() {
        for name, val := range inputs {
            p_list := g.inputs[name]     // Lookup this parameter in graph inputs
            for _, p := range p_list {  // Iterate through parameters in p_list
                f_block := p.FuncBlock
                f_name  := f_block.GetName()
                _, exists := f_queue[f_name]
                if !exists {
                    // Get the input parameters and create a ParamValues space for the function
                    f_ins, f_outs := f_block.GetParams()
                    all_inputs[f_name] = make(ParamValues, 0, cap(f_inputs))
                    all_inparams[f_name] = f_ins
                    all_outparams[f_name] = f_outs

                    // Add f_block to all_waiting
                    all_waiting[f_name] = f_block
                }

                all_inputs[f_name][p.GetName()] = val
            }
        }
    }

    // Iterate through all blocks that are waiting
    // to see if all of their inputs have been set.
    // If so, it makes it runs them.
    // Deleting them from waiting, and placing them in running.
    checkWaiting := func() {
        for f_name, f_block := range all_waiting {
            f_p := f_params[f_name]
            f_ins := f_inputs[f_name]
            switch {
                case len(f_p) == len(f_ins):
                    f_outs := make(chan ParamValues)
                    f_stop := make(chan bool)
                    f_err  := make(chan FlowError)
                    go f_block.Run(f_ins, f_outs, f_stop, f_err)
                    all_running[f_name] = f_block
                    all_outputs[f_name], all_stops[f_name], all_errs[fname] = f_outs, f_stop, f_err
                    delete(all_waiting[f_name])
                case len(f_p) < len(f_ins):
                    err <- FlowError{Ok: false, Info: "Parameter Collision", Addr: &g}
                default:
            }
        }
    }

    // Iterate through all blocks that are waiting
    // to see if all of their inputs have been set.
    // If so, it makes it runs them.
    // Deleting them from waiting, and placing them in running.
    checkRunning := func() {
      select {
          case f_return := <-data_out:                       // If an output is returned
              f_name := f_return.FuncBlock.GetName()         // Get the name of the block it belongs to
              f_outparams := all_outparams[f_name]           // Get the output parameters to check type
              if checkTypes(f_return.Values, f_outparams) {  // Check the types with output parameters
                  err <- FlowError {Ok: true}                // If good, return no error
                  outputs <- f_return                        // Along with the data
                  return                                     // And stop the function
              } else {
                  sendError("Outputs not the right type")
              }
          case <- stop:                             // If commanded to stop externally
              allStop()                             // Pass it on to subfunction
              return
          case temp_err := <-errs:                     // If there is an error, save it
              if !temp_err.Ok {
                  sendError(temp_err)
              }
      }
    }

    sendError := func(e interface{}) {
      switch E := e.(type) {
      case FlowError:
        err <- E
      case string:
        err <- FlowError{Ok: false, Info: E, Addr: &g}
      }
      allStop()
    }

    allStop := func() {
      for f_name, stop := range all_stops {
        stop <- true
        for {
          stopped := <-errs
          if stopped.Addr == all_running[f_name] {
              delete(all_running[f_name])
              break
          }
        }
      }
      err <- FlowError{Ok: false, Info: STOPPING, Addr: &g}
    }

}

// Checks if all keys in params are present in values
// And that all values are of their appropriate types as labeled in in params
func multiCheckTypes(values ParamValues, params MultiParamTypes) (ok bool) {
    var val interface{}
    for name, p_list := range params {
        for kind := range p_list {
            val, exists = values[name]
            switch x := val.(type) {
                case !exists:
                    return false
                case x != kind:
                    return false
            }
        }
    }
    return true
}

func checkTypes(values ParamValues, params ParamTypes) (ok bool) {
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