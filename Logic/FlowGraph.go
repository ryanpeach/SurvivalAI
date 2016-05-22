package graph

type Graph struct {
    name string
    nodes map[string]FunctionBlock
    edges map[FunctionBlock][]FunctionBlock
    inputs      map[string][]Parameter
    input_types map[string]Type
    outputs     map[string]Parameter
}

// Returns a copy of FunctionBlock's name
func (g Graph) GetName() string {return g.name}

// Returns copies of all parameters in FunctionBlock
func (g Graph) GetParams() (inputs []Parameter, outputs []Parameter) {
    inputs := make([]Parameter, len(g.inputs)),
    for name, p_list := range g.inputs {
        inputs.append(Parameter{FuncBlock: &g, ParamName: name, ParamType: p_list[0].ParamType})
    }

    outputs := make([]Parameter, len(g.outputs)),
    for name, p := range g.outputs {
        outputs.append(Parameter{FuncBlock: &g, ParamName: name, ParamType: p.ParamType})
    }
}

fun (g Graph) Run(inputs KeyValues,
                  outputs chan KeyValues,
                  stop chan bool,
                  err chan FlowError) {
    // Check types to ensure inputs are the type defined in input parameters
    if !checkTypes(inputs, g.inputs) {
      return FlowError{Ok: false, Info: "Inputs are impropper types.", Addr: &g}
    }

    // Declare variables
    all_waiting   := make(map[string]*FuncBlock)
    all_running   := make(map[string]*FuncBlock)
    all_inputs    := make(map[string]KeyValues)
    all_inparams  := make(map[string][]Parameter)
    all_outparams := make(map[string][]Parameter)
    all_stops     := make(map[string](chan bool))
    data_out      := make(chan DataValues)
    errs          := make(chan FlowError)

    // Iterates through all given inputs and adds them to method's all_inputs map
    // Also sets all_inparams, all_outparams, and all_blocks
    func init() {
        for name, val := range inputs {
            p_list := g.inputs[name]     // Lookup this parameter in graph inputs
            for _, p := range p_list {  // Iterate through parameters in p_list
                f_block := p.FuncBlock
                f_name  := f_block.GetName()
                _, exists := f_queue[f_name]
                if !exists {
                    // Get the input parameters and create a KeyValues space for the function
                    f_ins, f_outs := f_block.GetParams()
                    all_inputs[f_name] := make(KeyValues, 0, cap(f_inputs))
                    all_inparams[f_name] := f_ins
                    all_outparams[f_name] := f_outs

                    // Add f_block to all_waiting
                    all_waiting[f_name] := f_block
                }

                all_inputs[f_name][p.GetName()] := val
            }
        }
    }

    // Iterate through all blocks that are waiting
    // to see if all of their inputs have been set.
    // If so, it makes it runs them.
    // Deleting them from waiting, and placing them in running.
    func checkWaiting() {
        for f_name, f_block := range all_waiting {
            f_p := f_params[f_name]
            f_ins := f_inputs[f_name]
            switch {
                case len(f_p) == len(f_ins):
                    f_outs := make(chan KeyValues)
                    f_stop := make(chan bool)
                    f_err  := make(chan FlowError)
                    go f_block.Run(f_ins, f_outs, f_stop, f_err)
                    all_running[f_name] := f_block
                    all_outputs[f_name], all_stops[f_name], all_errs[fname] := f_outs, f_stop, f_err
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
    func checkRunning() {
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

    func sendError(e interface{}) {
      switch E := e.(type) {
      case FlowError:
        err <- E
      case string:
        err <- FlowError{Ok: false, Info: E, Addr: &g}
      }
      allStop()
    }

    func allStop() {
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

func (g Graph) New(label string, m1 *Method, n1 string, m2 *Method, n2 string) (ok bool) {

}

// Checks if all keys in params are present in values
// And that all values are of their appropriate types as labeled in in params
func checkTypes(values KeyValues, params map[string]Type) (ok bool) {
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

func (g Graph) Draw() (ok bool) {

}
