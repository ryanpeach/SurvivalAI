package primitives

const {
  STOPPING := "STOP"
}

type FlowError struct{
    Ok bool
    Info string
    Addr *FunctionBlock
}

type Parameter struct {
    funcBlock *FunctionBlock
    paramName string
    paramType Type
}
func (p Parameter) GetName() string {return p.paramName}
func (p Parameter) GetType() Type   {return p.paramType}
func (p Parameter) GetBlock() *FunctionBlock {return p.funcBlock}

type DataOut struct {
    FuncBlock *FunctionBlock
    Values KeyValues
}

type KeyValues map[string]interface{}
type DataStream func(inputs KeyValues,
                     outputs chan DataOut,
                     stop chan bool
                     err chan FlowError)

type FunctionBlock interface{
    Run DataStream
    GetName() string
    GetParams() []Parameter
}
// A primitive function block that only
// contains a DataStream Function to run
type PrimitiveBlock struct {
    name    string
    fn      DataStream
    inputs  map[string]Type
    outputs map[string]Type
}

// Returns a copy of FunctionBlock's name
func (m PrimitiveBlock) GetName() string {return m.name}

// Returns copies of all parameters in FunctionBlock
func (m PrimitiveBlock) GetParams() (inputs []Parameter, outputs []Parameters) {
    inputs := make([]Parameter, len(m.inputs)),
    for name, t := range m.inputs {
      inputs.append(Parameter{funcBlock: &m, paramName: name, paramType: t})
    }

    outputs := make([]Parameter, len(m.outputs))
    for name, t := range m.outputs {
      outputs.append(Parameter{funcBlock: &m, paramName: name, paramType: t})
    }
    return
}

// Run the function
fun (m PrimitiveBlock) Run(inputs KeyValues,
                           outputs chan DataOut,
                           stop chan bool,
                           err chan FlowError) {
    // Check types to ensure inputs are the type defined in input parameters
    if !checkTypes(inputs, m.inputs) {
      return FlowError{Ok: false, Info: "Inputs are impropper types.", Addr: &m}
    }

    // Duplicate the given channel to pass to the enclosed function
    // Run the function
    f_err  := make(chan FlowError)
    f_out  := make(chan DataOut)
    f_stop := make(chan bool)
    go m.fn(inputs, f_out, f_stop, f_err)

    // Wait for a stop or an output
    var temp_err FlowError
    for {
        select {
            case f_return := <-f_out:                 // If an output is returned
                if checkTypes(f_return, m.outputs) {  // Check the types with output parameters
                    err <- FlowError {Ok: true}       // If good, return no error
                    outputs <- DataOut{&m, f_return}  // Along with the data
                    return                            // And stop the function
                }
            case <-stop:                              // If commanded to stop externally
                f_stop <- true                        // Pass it on to subfunction
                return                                // And stop immediately
            case temp_err = f_err:                    // If there is an error, save it
                if !temp_err.Ok {                     // See if it is bad
                    err <- temp_err                   // If it is bad, pass it up the chain
                    return                            // And stop the function
                }
        }
    }
}

// Initializes a FunctionBlock object with given attributes, and an empty parameter list.
// The only way to create Methods's
func New(name string, function DataStream, inputs map[string]Type, outputs map[string]Type) FunctionBlock {
    return PrimitiveBlock{name: name,
                          fn: function,
                          inputs: inputs,
                          outputs: outputs}
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
