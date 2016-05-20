package Logic

type Process template {
    SetValue()
    GetValue()
    Run()
}

type DataFlow struct {
    Name string
    output *Param
    input  []*Param
}

type Graph struct {
    Nodes map[string]*Method
    Edges map[string]DataFlow
}

func (g Graph) NewDataFlow(label string, m1 *Method, n1 string, m2 *Method, n2 string) (ok bool) {
    
}

func (f DataFlow) PassData() (ok bool) {
    val, ok := f.output.GetValue()
    for p := range f.input {
        if !ok {return false}       // Handle errors
        _, ok = p.SetValue(val)     // Set each value for each input
    }
    return f.output.Reset()         // Reset the output and return the status
}

func (g Graph) Draw() (ok bool) {
    
}