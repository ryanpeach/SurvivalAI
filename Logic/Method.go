package Logic

import "type/Type"

// A function block that contains parameters, a name, and a function that operates on the parameters.
type Method struct {
    Name   string
    params map[string]*Param
    F      func(map[string]*Param)
}

// The Param (short for Parameter) type is a required type for Methods
// They have a name, and a type. The val parameter is private.
// They also contain all their relevant information as to their place within the method.
// These structures should be initialized with the function NewParameter.
type Param struct {
    Is_Input bool
    T        Type
    val      interface{}
}

// Checks the type of newval and sets Param.val to newval if compatible.
// Returns the value of Param as well as a bool representing whether or not it was changed.
func (p Param) SetValue(newval interface{}) (p.val interface{}, ok bool) {
    switch vT := newval.(type) {
        case p.T:
            p.val = newval
            ok := true
        default:
            ok := false
    }
    return
}

// Returns the value of Param (Param.val).
func (p Param) GetValue() interface{} {return p.val}

// Initializes a Param object within Method with given attributes, and a nil value (Param.val).
// The only way to create Param's
func (m Method) AddParameter(name string, is_input bool, t Type) (Param, bool) {
    // Check if name already in Method
    val, exists := m.params[name]
    if exists {ok := false}  // Return a pointer to the Param and false to indicate no Param created
    else {
        // Otherwise, create the parameter, and add it to the map
        val = &Param{Is_Input: is_input, T: t, val: nil}
        m.params[name] = val
        ok := true
    }
    return *val, ok
}

// Returns copies of all parameters in Method
func (m Method) GetParameters() map[string]Param {
    out := make(map[string]Param, cap(m.params))
    for key, val := range m.params {
        out[key] = *val
    }
    return out
}

// Initializes a Method object with given attributes, and an empty parameter list.
// The only way to create Methods's
func NewMethod(Name string, f func(map[string]*Param)) (val *Method, ok bool) {
    return Method{Name: name,
                  param: make(map[string]*Param, n_params)
                  F: f}
}