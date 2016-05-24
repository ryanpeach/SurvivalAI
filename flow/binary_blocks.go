package flow

import "time"

// Creates a variety of blocks for paired operations
func opGeneral(id InstanceID, t TypeStr, outname string, opfunc func(in ParamValues, out *ParamValues)) FunctionBlock {
    // Create Plus block
    ins := ParamTypes{"A": t, "B": t}
    outs := ParamTypes{"OUT": t}
    
    // Define the function as a closure
    runfunc := func(inputs ParamValues,
                     outputs chan DataOut,
                     stop chan bool,
                     err chan FlowError) {
        data := make(ParamValues)
        opfunc(inputs, &data)
        out := DataOut{BlockID: id, Values: data}
        outputs <- out
        return
    }
    
    // Initialize the block and return
    return PrimitiveBlock{name: outname, fn: runfunc, id: id, inputs: ins, outputs: outs}
}

// Numeric Float Functions
func PlusFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) + in["B"].(float64)
    }
    name := "numeric_plus_float"
    return opGeneral(id,"float",name,opfunc)
}
func SubFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) - in["B"].(float64)
    }
    name := "numeric_subtract_float"
    return opGeneral(id,"float",name,opfunc)
}
func MultFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) * in["B"].(float64)
    }
    name := "numeric_multiply_float"
    return opGeneral(id,"float",name,opfunc)
}
func DivFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) / in["B"].(float64)
    }
    name := "numeric_divide_float"
    return opGeneral(id,"float",name,opfunc)
}

// Numeric Int Functions
func PlusInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) + in["B"].(int)
    }
    name := "numeric_plus_int"
    return opGeneral(id,"int",name,opfunc)
}
func SubInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) - in["B"].(int)
    }
    name := "numeric_subtract_int"
    return opGeneral(id,"int",name,opfunc)
}
func MultInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) * in["B"].(int)
    }
    name := "numeric_multiply_int"
    return opGeneral(id,"int",name,opfunc)
}
func DivInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) / in["B"].(int)
    }
    name := "numeric_divide_int"
    return opGeneral(id,"int",name,opfunc)
}
func Mod(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) % in["B"].(int)
    }
    name := "numeric_mod_int"
    return opGeneral(id,"int",name,opfunc)
}

// Boolean Logic Functions
func And(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) && in["B"].(bool)
    }
    name := "logical_and"
    return opGeneral(id,"bool",name,opfunc)
}
func Or(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) || in["B"].(bool)
    }
    name := "logical_or"
    return opGeneral(id,"bool",name,opfunc)
}
func Xor(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) != in["B"].(bool)
    }
    name := "logical_xor"
    return opGeneral(id,"bool",name,opfunc)
}
func Nand(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = !(in["A"].(bool) && in["B"].(bool))
    }
    name := "logical_nand"
    return opGeneral(id,"bool",name,opfunc)
}
func Nor(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = !(in["A"].(bool) || in["B"].(bool))
    }
    name := "logical_nor"
    return opGeneral(id,"bool",name,opfunc)
}