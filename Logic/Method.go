package Logic

type Parameters map[string]Type
type KeyValues map[string]interface{}

type FunctionBlock interface {
    Run() func(inputs KeyValues,
               outputs chan KeyValues,
               stop chan bool)
    GetParams() (inputs Parameters, outputs Parameters)
    GetName() string
}

// A function block that contains parameters, a name, and a function that operates on the parameters.
type functionBlock struct {
    name    string
    F       
    inputs  Parameters
    outputs Parameters
}

// Run the method function
fun (m functionBlock) Run(inputs KeyValues, outputs chan KeyValues, stop chan bool) {
    if CheckTypes(inputs, m.inputs) {
        F_out = make(chan KeyValues)
        F_stop = make(chan bool)
        go m.F(inputs, F_out, F_stop)
        select {
            case temp_out <- F_out:
                if CheckTypes(temp_out, m.outputs) {
                    outputs <- temp_out
                }
            case <-stop:
                F_stop <- true
                temp <- F_out   // Wait for F to return
                return
        }
    }
}

// Returns copies of all parameters in FunctionBlock
func (f functionBlock) GetParams() (inputs Parameters, outputs Parameters) {
    inputs := m.inputs
    outputs := m.outputs
    return
}

// Returns a copy of FunctionBlock's name
func (f functionBlock) GetName() string {
    return m.name
}

// Initializes a FunctionBlock object with given attributes, and an empty parameter list.
// The only way to create Methods's
func NewFunctionBlock(name string, inputs Parameters, outputs Parameters) FunctionBlock {
    return functionBlock{name: name,
                         inputs: inputs,
                         outputs: outputs}
}

// Checks if all keys in params are present in values
// And that all values are of their appropriate types as labeled in in params
func CheckTypes(values KeyValues, params Parameters) (ok bool) {
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