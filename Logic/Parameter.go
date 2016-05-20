package Logic

import "type/Type"

type Parameter interface {
    SetValue(interface{}) (interface{}, bool)
    GetValue() interface{}
}

type Param struct {
    Name     string
    Is_Input bool
    T        Type
    Parent   *Method
    val      interface{}
}

func (p Param) SetValue(newval interface{}) {
    switch vT := newval.(type) {
        case p.T:
            p.val = newval
            return (p.val, true)
        default:
            return (p.val, false)
    }
}
func (p Param) GetValue() interface{} {return p.val}

func NewParameter(name string, is_input bool, t Type, p *Method) Param {
    return Param{Name: name, Is_Input: is_input, T: t, Parent: p, val: nil}
}