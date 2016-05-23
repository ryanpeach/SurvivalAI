package flow

const (
  STOPPING = "STOP" // Used to declare a stopping error
)

// Used to declare an error in the flow pipeline
type FlowError struct{
    Ok bool
    Info string
    Addr *FunctionBlock
}

// Used to represent a parameter to a FunctionBlock
// Everything is private, as this struct is immutable
type Parameter struct {
    blockID   InstanceID
    paramName string
    paramType TypeStr
}
func (p Parameter) GetName() string {return p.paramName}
func (p Parameter) GetType() TypeStr {return p.paramType}
func (p Parameter) GetBlock() *FunctionBlock {return p.funcBlock}

// Used to store the outputs of a FunctionBlock, while keeping it's reference.
type DataOut struct {
    BlockID InstanceID
    Values  ParamValues
}

// KeyValues and DataStreams are the types of values and functions
// Used universally inside FunctionBlocks
type TypeStr string
type InstanceID int
type ParamValues map[string]interface{}
type ParamTypes map[string]TypeStr
type DataStream func(inputs ParamValues,
                     outputs chan DataOut,
                     stop chan bool,
                     err chan FlowError)

// The primary interface of the flowchart. Allows running, has a name, and has parameters.
type FunctionBlock interface{
    Run(inputs ParamValues,
        outputs chan DataOut,
        stop chan bool,
        err chan FlowError)
    GetName() string
    GetParams() (inputs []Parameter, outputs []Parameter)
}

// A primitive function block that only
// contains a DataStream Function to run
type PrimitiveBlock struct {
    name    string
    id      InstanceID
    fn      DataStream
    inputs  ParamTypes
    outputs ParamTypes
}

// Returns a copy of FunctionBlock's name
func (m PrimitiveBlock) GetName() string {return m.name}

// Returns copies of all parameters in FunctionBlock
func (m PrimitiveBlock) GetParams() (inputs []Parameter, outputs []Parameter) {
    inputs = make([]Parameter, len(m.inputs))
    for name, t := range m.inputs {
      append(inputs, Parameter{funcBlock: &m, paramName: name, paramType: t})
    }

    outputs = make([]Parameter, len(m.outputs))
    for name, t := range m.outputs {
      append(outputs, Parameter{funcBlock: &m, paramName: name, paramType: t})
    }
    return
}

// Run the function
func (m PrimitiveBlock) Run(inputs ParamValues,
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
                if checkTypes(f_return.Values, m.outputs) {  // Check the types with output parameters
                    err <- FlowError {Ok: true}       // If good, return no error
                    outputs <- DataOut{&m, f_return.Values}  // Along with the data
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
func New(name string, function DataStream, inputs ParamTypes, outputs ParamTypes) FunctionBlock {
    return PrimitiveBlock{name: name,
                          fn: function,
                          inputs: inputs,
                          outputs: outputs}
}

// Checks if all keys in params are present in values
// And that all values are of their appropriate types as labeled in in params
func checkTypesPrimitive(values ParamValues, params ParamTypes) (ok bool) {
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
